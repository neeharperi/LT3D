# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class CBGS(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CBGS,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)

    @property
    def with_velocity(self):
        """bool: Whether the head predicts velocity"""
        return self.pts_bbox_head is not None and \
            self.pts_bbox_head.with_velocity

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)
        return merged_bboxes

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]
