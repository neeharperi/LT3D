# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import tempfile
import warnings
from os import path as osp
from pathlib import Path
from turtle import width

import mmcv
import numpy as np
import pandas as pd
from av2.geometry.camera.pinhole_camera import Intrinsics, PinholeCamera
from av2.geometry.geometry import quat_to_mat
from av2.geometry.se3 import SE3
from av2.structures.cuboid import Cuboid
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import torch

from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from ..core import show_multi_modality_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes,
                         LiDARInstance3DBoxes, get_box_type)
from .builder import DATASETS
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline


def yaw_to_quaternion3d(yaw: float) -> np.ndarray:
    """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.
    Args:
        yaw: angle to rotate about the z-axis, representing an Euler angle, in radians
    Returns:
        array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    """
    qx, qy, qz, qw = Rotation.from_euler(seq="z", angles=yaw, degrees=False).as_quat()
    return np.array([qw, qx, qy, qz])

def get_cam_model(img_info):
    ego_SE3_cam = SE3(rotation=np.array(img_info["ego_SE3_cam_rotation"]), translation=np.array(img_info["ego_SE3_cam_translation"]))
    K = np.array(img_info["ego_SE3_cam_intrinsics"])
    fx_px = K[0, 0]
    fy_px = K[1, 1]
    cx_px = K[0, 2]
    cy_px = K[1, 2]
    width = img_info["width"]
    height = img_info["height"]
    intrinsics = Intrinsics(fx_px, fy_px, cx_px, cy_px, width, height)
    cam_model = PinholeCamera(ego_SE3_cam=ego_SE3_cam, intrinsics=intrinsics, cam_name=img_info["id"])

    return cam_model

@DATASETS.register_module()
class AV2MonoDataset(Dataset):
    CLASSES = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 img_prefix='',
                 box_type_3d='Camera',
                 filter_empty_gt=True,
                 test_mode=False,
                 file_client_args=dict(backend='disk')):
        super().__init__()
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.bbox_code_size = 9
        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # load annotations
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(open(local_path, 'rb'))
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos["annotations"][idx]
        gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.
        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return mmcv.load(ann_file, file_format='json')

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:
                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos["images"][index]
        sample_idx = info['token']
  
        img_info = dict(
            filename=info['file_name'],
            sample_idx=sample_idx,
            id=info["id"],
            size=(info['width'], info['height']),
            cam_intrinsic=info['ego_SE3_cam_intrinsics'],
            transform={"ego_SE3_cam_rotation": info['ego_SE3_cam_rotation'],
                       "ego_SE3_cam_translation": info['ego_SE3_cam_translation'],
                       "ego_SE3_cam_intrinsics": info['ego_SE3_cam_intrinsics']},
            )

        input_dict = dict(img_info=img_info)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.
        Args:
            index (int): Index of the annotation data to get.
        Returns:
            dict: Annotation information consists of the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos["annotations"][index]
        img_info = self.data_infos["images"][index]

        ego_SE3_cam = SE3(rotation=np.array(img_info["ego_SE3_cam_rotation"]), translation=np.array(img_info["ego_SE3_cam_translation"]))
   
        gt_bboxes = []
        gt_labels = []
        attr_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []

        for i, ann in enumerate(info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2id[ann['category_name']])
                attr_labels.append(ann['attribute_id'])
                gt_masks_ann.append(ann.get('segmentation', None))
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                velo_cam3d = np.array(ann['velo_cam3d']).reshape(1, 2)
                nan_mask = np.isnan(velo_cam3d[:, 0])
                velo_cam3d[nan_mask] = [0.0, 0.0]
                bbox_cam3d = np.concatenate([bbox_cam3d, velo_cam3d], axis=-1)
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze())
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            attr_labels = np.array(attr_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            attr_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        gt_bboxes_cam3d = LiDARInstance3DBoxes(gt_bboxes_cam3d, 
                                               box_dim=gt_bboxes_cam3d.shape[-1]).convert_to(Box3DMode.CAM, ego_SE3_cam.inverse().transform_matrix)
            
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['file_name']

        anns_results = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_labels_3d=gt_labels_3d,
            attr_labels=attr_labels,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return anns_results

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_prefix'] = self.img_prefix
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def nms(self, predictionsDataFrame):
        predictions = {"tx_m" : [],
                       "ty_m" : [],
                       "tz_m" : [],
                       "length_m" : [],
                       "width_m" : [],
                       "height_m" : [],
                       "qw" : [],
                       "qx" : [],
                       "qy" : [],
                       "qz" : [],
                       "vx" : [],
                       "vy" : [],
                       "score" : [],
                       "log_id" : [],
                       "timestamp_ns" : [],
                       "category" : []}

        sweeps = set()

        for rowTable in predictionsDataFrame.iterrows():
            log_id = rowTable[1]["log_id"]
            timestamp = rowTable[1]["timestamp_ns"]

            sweeps.add((log_id, timestamp))

        for log_id, timestamp in mmcv.track_iter_progress(sweeps):
            sweep_preds = predictionsDataFrame[predictionsDataFrame["log_id"] == log_id]
            sweep_preds = sweep_preds[sweep_preds["timestamp_ns"] == timestamp]

            boxes_3d = []
            scores = []
            labels = []

            for rowTable in sweep_preds.iterrows():
                translation = [rowTable[1]["tx_m"], rowTable[1]["ty_m"], rowTable[1]["tz_m"]]
                lwh = [rowTable[1]["length_m"], rowTable[1]["width_m"], rowTable[1]["height_m"]]
                rotation = quat_to_mat(np.array([rowTable[1]["qw"], rowTable[1]["qx"], rowTable[1]["qy"], rowTable[1]["qz"]]))
                ego_SE3_object = SE3(rotation=rotation, translation=np.array(translation))
                rot = ego_SE3_object.rotation
                translation[2] = translation[2] - lwh[2] / 2
                yaw = [math.atan2(rot[1, 0], rot[0, 0])]

                vel = [rowTable[1]["vx"], rowTable[1]["vy"]]
                score = rowTable[1]["score"]
                label = self.CLASSES.index(rowTable[1]["category"])
                
                boxes_3d.append(translation + lwh + yaw + vel)
                scores.append(score)
                labels.append(label)

            boxes_3d = torch.tensor(boxes_3d).cuda()
            cam_boxes3d = LiDARInstance3DBoxes(boxes_3d, box_dim=9)
            scores = torch.Tensor(scores).cuda()
            labels = torch.LongTensor(labels).cuda()
            nms_scores = scores.new_zeros(scores.shape[0], 26 + 1)
            indices = labels.new_tensor(list(range(scores.shape[0])))
            nms_scores[indices, labels] = scores
                
            # box nms 3d over 6 images in a frame
            # TODO: move this global setting into config
            nms_cfg = dict(
                use_rotate_nms=True,
                nms_across_levels=False,
                nms_pre=4096,
                nms_thr=0.05,
                score_thr=0.01,
                min_bbox_size=0,
                max_per_frame=500)
            from mmcv import Config
            nms_cfg = Config(nms_cfg)
            cam_boxes3d_for_nms = xywhr2xyxyr(cam_boxes3d.bev)
            boxes3d = cam_boxes3d.tensor
            # generate attr scores from attr labels
            attrs = 0 * labels
            boxes3d, scores, labels, attrs = box3d_multiclass_nms(
                boxes3d,
                cam_boxes3d_for_nms,
                nms_scores,
                nms_cfg.score_thr,
                nms_cfg.max_per_frame,
                nms_cfg,
                mlvl_attr_scores=attrs)
            
            for gt_bboxes_3d, score, label in zip(boxes3d.cpu(), scores.cpu(), labels.cpu()):
                center = gt_bboxes_3d[:3].numpy()
                center[2] = gt_bboxes_3d[2] + gt_bboxes_3d[5] / 2
                quat = yaw_to_quaternion3d(gt_bboxes_3d[6])
                lwh = gt_bboxes_3d[3:6].numpy()
                velocity = gt_bboxes_3d[6:8].numpy()

                predictions['tx_m'].append(center[0])
                predictions['ty_m'].append(center[1])
                predictions['tz_m'].append(center[2])

                predictions['qw'].append(quat[0])
                predictions['qx'].append(quat[1])
                predictions['qy'].append(quat[2])
                predictions['qz'].append(quat[3])

                predictions['length_m'].append(lwh[0])
                predictions['width_m'].append(lwh[1])
                predictions['height_m'].append(lwh[2])

                predictions['vx'].append(velocity[0])
                predictions['vy'].append(velocity[1])

                predictions['category'].append(self.CLASSES[label.item()])
                predictions['score'].append(score.item())

                predictions['log_id'].append(log_id)
                predictions['timestamp_ns'].append(timestamp)

        predictionsDataFrame = pd.DataFrame.from_dict(predictions)
        return predictionsDataFrame

    def format_results(self, outputs):
        """Format the results to be recognizable to the Argoverse eval script.
        Args:
            outputs (list[dict]): Testing results of the dataset.
            dt_root (str):
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """

        predictions = {"tx_m" : [],
                       "ty_m" : [],
                       "tz_m" : [],
                       "length_m" : [],
                       "width_m" : [],
                       "height_m" : [],
                       "qw" : [],
                       "qx" : [],
                       "qy" : [],
                       "qz" : [],
                       "vx" : [],
                       "vy" : [],
                       "score" : [],
                       "log_id" : [],
                       "timestamp_ns" : [],
                       "category" : []}

        ground_truth = {"tx_m" : [],
                       "ty_m" : [],
                       "tz_m" : [],
                       "length_m" : [],
                       "width_m" : [],
                       "height_m" : [],
                       "qw" : [],
                       "qx" : [],
                       "qy" : [],
                       "qz" : [],
                       "vx" : [],
                       "vy" : [],
                       "score" : [],
                       "num_interior_pts" : [],
                       "log_id" : [],
                       "track_uuid" : [],
                       "timestamp_ns" : [],
                       "category" : []}

        for index, pred_dict in enumerate(mmcv.track_iter_progress(outputs)):
            img_info = self.data_infos["images"][index]
            anno_info = self.data_infos["annotations"][index]
            log_id = img_info["token"]
            timestamp_ns = img_info["timestamp"]

            cam_model = get_cam_model(img_info)
            ego_SE3_cam = cam_model.ego_SE3_cam
            
            for anno in anno_info:
                bbox_cam3d = anno["bbox_cam3d"]
                velo_cam3d = anno["velo_cam3d"]
                cat_name = anno["category_name"]
                uuid = anno["gt_uuid"]
                num_pts = anno["gt_num_pts"]

                bboxes_cam3d = np.expand_dims(np.concatenate([bbox_cam3d, velo_cam3d], axis=-1), axis=0)
                gt_bboxes_3d = LiDARInstance3DBoxes(bboxes_cam3d, box_dim=bboxes_cam3d.shape[-1]).tensor[0]
            
                velocity = ego_SE3_cam.transform_from([*velo_cam3d, 0])[:2]

                center = gt_bboxes_3d[:3].numpy()
                lwh = gt_bboxes_3d[3:6].numpy()
                quat = yaw_to_quaternion3d(gt_bboxes_3d[6])

                ground_truth['tx_m'].append(center[0])
                ground_truth['ty_m'].append(center[1])
                ground_truth['tz_m'].append(center[2] + lwh[2] / 2)

                ground_truth['qw'].append(quat[0])
                ground_truth['qx'].append(quat[1])
                ground_truth['qy'].append(quat[2])
                ground_truth['qz'].append(quat[3])

                ground_truth['length_m'].append(lwh[0])
                ground_truth['width_m'].append(lwh[1])
                ground_truth['height_m'].append(lwh[2])

                ground_truth['vx'].append(velocity[0])
                ground_truth['vy'].append(velocity[1])

                ground_truth['category'].append(cat_name)
                ground_truth['score'].append(-1)
                ground_truth['num_interior_pts'].append(num_pts)
                ground_truth['track_uuid'].append(uuid)

                ground_truth['log_id'].append(log_id)
                ground_truth['timestamp_ns'].append(timestamp_ns)


            lidar_bboxes = pred_dict['img_bbox']['boxes_3d'].convert_to(Box3DMode.LIDAR, ego_SE3_cam.transform_matrix)
            for bbox, score, label in zip(
                lidar_bboxes,
                pred_dict['img_bbox']['scores_3d'],
                pred_dict['img_bbox']['labels_3d'],
            ):
                gt_bboxes_3d = bbox[0:7]
                velo_cam3d = bbox[7:9]
                velocity = ego_SE3_cam.transform_from([*velo_cam3d, 0])[:2]

                center = gt_bboxes_3d[:3].numpy()
                lwh = gt_bboxes_3d[3:6].numpy()
                quat = yaw_to_quaternion3d(gt_bboxes_3d[6])

                predictions['tx_m'].append(center[0])
                predictions['ty_m'].append(center[1])
                predictions['tz_m'].append(center[2] + lwh[2] / 2)

                predictions['qw'].append(quat[0])
                predictions['qx'].append(quat[1])
                predictions['qy'].append(quat[2])
                predictions['qz'].append(quat[3])

                predictions['length_m'].append(lwh[0])
                predictions['width_m'].append(lwh[1])
                predictions['height_m'].append(lwh[2])

                predictions['vx'].append(velocity[0])
                predictions['vy'].append(velocity[1])

                predictions['category'].append(self.CLASSES[label.item()])
                predictions['score'].append(score.item())

                predictions['log_id'].append(log_id)
                predictions['timestamp_ns'].append(timestamp_ns)

        predictionsDataFrame = pd.DataFrame.from_dict(predictions)
        groundTruthDataFrame = pd.DataFrame.from_dict(ground_truth)

        predictionsDataFrame = self.nms(predictionsDataFrame)
        return predictionsDataFrame, groundTruthDataFrame

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        print("Visualize Detections")

        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(mmcv.track_iter_progress(results[:20])):
            if 'img_bbox' in result.keys():
                result = result['img_bbox']

            data_info = self.data_infos["images"][i]
            img_path = data_info['file_name']
            file_name = data_info["token"] + "-" + osp.split(img_path)[-1].split('.')[0]

            img, _ = self._extract_data(i, pipeline, ['img', 'img_metas'])
            intrinsics = np.array(data_info["ego_SE3_cam_intrinsics"])
            # need to transpose channel to first dim
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            pred_bboxes = result['boxes_3d']

            show_multi_modality_result(
                img,
                gt_bboxes,
                pred_bboxes,
                intrinsics,
                out_dir,
                file_name,
                box_mode='camera',
                show=show)

    def evaluate(self, results, out_path=None, **kwargs):
        """Evaluate.
        """
        from av2.evaluation.detection.eval import evaluate
        from av2.evaluation.detection.utils import DetectionCfg

        split = "val"
        if out_path is not None:
            pipeline = kwargs.get("pipeline", None)
            self.show(results, out_path + "/visuals/", show=False, pipeline=pipeline)

        predictionsDataFrame, groundTruthDataFrame = self.format_results(results)

        if out_path is not None:
            pd.DataFrame.to_csv(groundTruthDataFrame, out_path + "/{}_gt.csv".format(split))
            pd.DataFrame.to_csv(predictionsDataFrame, out_path + "/{}_detections.csv".format(split))

        metric_type = kwargs.get("metric_type", None)
        max_range = 50
       
        cfg = DetectionCfg(dataset_dir = Path("/home/ubuntu/Workspace/Data/Sensor/{}".format(split)), max_range_m=max_range)

        _, _, metrics = evaluate(predictionsDataFrame, groundTruthDataFrame, cfg)
        print(metrics)
        
        if out_path is not None:
            max_range_tag = "_{}m".format(max_range)
            metric_tag = "_{}".format(metric_type)
            pd.DataFrame.to_csv(metrics, out_path + "/results{}{}.csv".format(max_range_tag, metric_tag))

        return metrics.to_json()

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('_build_default_pipeline is not implemented '
                                  f'for dataset {self.__class__.__name__}')

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.
        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.
        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.
        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.
        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos["annotations"])

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.
        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
