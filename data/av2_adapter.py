import math
import os
from copy import deepcopy
from typing import Dict, List
from uuid import UUID

import cv2
import numpy as np
from av2.datasets.sensor.sensor_dataloader import read_city_SE3_ego
from av2.geometry.se3 import SE3
from pyquaternion import Quaternion
from shapely import affinity
from shapely.errors import TopologicalError
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseMultipartGeometry
from pathlib import Path
from data.utils import *


def av2_load_predictions_and_labels(
    prediction_path: str,
    infos_path: str,
    dataset_dir: str,
):
    classes = AV2_CLASS_NAMES
    raw_predictions = load(prediction_path)
    raw_labels = load(infos_path)
    prediction_list = unpack_predictions(raw_predictions, classes)
    label_list = unpack_labels(raw_labels)
    annotate_frame_metadata(prediction_list, label_list, ["seq_id", "timestamp_ns"])
    predictions = group_frames(prediction_list)
    labels = group_frames(label_list)
    city_SE3_by_seq_id = read_city_SE3_ego_by_seq_id(
       dataset_dir, seq_ids=predictions.keys()
    )
    predictions = transform_to_global_reference(predictions, city_SE3_by_seq_id)
    labels = transform_to_global_reference(labels, city_SE3_by_seq_id)
    return predictions, labels


def load_labels_from_dataframe(df, dataset_dir, downsample_rate):
    """
    Convert annotation dataframe to labels
    """
    classes = AV2_CLASS_NAMES
    df = df.drop_duplicates()
    label_list = []
    groups = df.groupby(["log_id", "timestamp_ns"])
    for (seq_id, timestamp_ns), frame_df in progressbar(groups, total=len(groups)):
        frame_df = frame_df[
            frame_df["category"].isin(classes) & (frame_df["num_interior_pts"] > 0)
        ]
        quaternions = frame_df[["qw", "qx", "qy", "qz"]].to_numpy()
        yaws = np.array([Quaternion(qs).yaw_pitch_roll[0] for qs in quaternions])
        translation = frame_df[["tx_m", "ty_m", "tz_m"]].to_numpy()
        size = frame_df[["length_m", "width_m", "height_m"]].to_numpy()
        translation[:, 2] -= size[:, 2] / 2
        label_list.append(
            {
                "translation_m": translation,
                "size": size,
                "yaw": yaws,
                "velocity_m_per_s": np.zeros((len(frame_df), 2)),  # doesn't exist
                "label": frame_df["category"]
                .map(lambda c: classes.index(c))
                .to_numpy(),
                "name": frame_df["category"].to_numpy(),
                "track_id": frame_df["track_uuid"]
                .map(lambda id: UUID(id).int)
                .to_numpy(),
                "timestamp_ns": timestamp_ns,
                "seq_id": seq_id,
            }
        )
    labels = group_frames(label_list)
    labels = {seq_id: frames[::downsample_rate] for seq_id, frames in labels.items()}
    city_SE3_by_seq_id = read_city_SE3_ego_by_seq_id(dataset_dir, seq_ids=labels.keys())
    labels = transform_to_global_reference(labels, city_SE3_by_seq_id)
    return labels


def unpack_labels(labels: List[Dict]) -> List[Dict]:
    """
    Returns
    -------
    list:
    """
    unpacked_labels = []
    for label in labels:
        bboxes = (
            np.array(label["gt_bboxes"])
            if len(label["gt_bboxes"]) > 0
            else np.zeros((0, 7))
        )
        velocity = (
            np.array(label["gt_velocity"])
            if len(label["gt_velocity"]) > 0
            else np.zeros((0, 2))
        )
        unpacked_labels.append(
            {
                "translation_m": bboxes[:, :3],
                "size": bboxes[:, 3:6],
                "yaw": wrap_pi(bboxes[:, 6]),
                "velocity_m_per_s": velocity,
                "label": np.array(label["gt_labels"], dtype=int),
                "name": np.array(label["gt_names"]),
                "track_id": np.array([UUID(id).int for id in label["gt_uuid"]]),
                "timestamp_ns": label["timestamp"],
                "seq_id": label["log_id"],
            }
        )
    return unpacked_labels


def read_city_SE3_ego_by_seq_id(
    dataset_dir: str, seq_ids: List[str]
) -> Dict[str, Dict[int, SE3]]:
    return {
        seq_id: read_city_SE3_ego(Path(os.path.join(dataset_dir, seq_id)))
        for seq_id in seq_ids
    }


def transform_to_global_reference(
    detections: Dict[str, List[Dict]], city_SE3_by_seq_id: Dict[str, Dict[int, SE3]]
):
    detections = deepcopy(detections)
    for seq_id, frames in detections.items():
        for detection in frames:
            # transform xyz, velocity and yaw to city reference frame
            ego_to_city_SE3 = city_SE3_by_seq_id[seq_id][detection["timestamp_ns"]]
            detection["translation_m"] = ego_to_city_SE3.transform_from(
                detection["translation_m"]
            )  # I (number of instances), 3
            detection["ego_translation_m"] = list(
                ego_to_city_SE3.transform_from(np.zeros(3))
            )
            rotation = ego_to_city_SE3.rotation
            velocity_3d = np.pad(
                detection["velocity_m_per_s"], [(0, 0), (0, 1)]
            )  # pad last dimension -> [x, y, 0]
            detection["velocity_m_per_s"] = velocity_3d @ rotation.T  # I, 3
            ego_to_city_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            detection["yaw"] = wrap_pi(detection["yaw"] + ego_to_city_yaw)  # I

    return detections


def transform_to_ego_reference(
    detections: Dict[str, List[Dict]], city_SE3_by_seq_id: Dict[str, Dict[int, SE3]]
):
    detections = deepcopy(detections)
    for seq_id, frames in detections.items():
        for detection in frames:
            # transform xyz, velocity and yaw to ego reference frame
            city_to_ego_SE3 = city_SE3_by_seq_id[seq_id][
                detection["timestamp_ns"]
            ].inverse()
            detection["translation_m"] = city_to_ego_SE3.transform_from(
                detection["translation_m"]
            )  # I (number of instances), 3
            rotation = city_to_ego_SE3.rotation
            detection["velocity_m_per_s"] = (detection["velocity_m_per_s"] @ rotation.T)[:, :2]  # I, 2
            ego_to_city_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            detection["yaw"] = wrap_pi(detection["yaw"] + ego_to_city_yaw)  # I

    return detections


def map_patch_to_canvas(
    geometry, patch_corner: np.ndarray, patch_size: np.ndarray, canvas_size: np.ndarray
) -> List[np.ndarray]:
    """
    polygon, patch_corner: x, y
    patch_size, canvas_size: h, w (y_axis, x_axis)
    returns int32 ndarray
    """
    patch_size, canvas_size = np.flip(patch_size), np.flip(canvas_size)  # map to x, y
    canvas = Polygon([(0, 0), (0, canvas_size[1]), canvas_size, (canvas_size[0], 0)])
    geometry = affinity.translate(geometry, *(-patch_corner))
    geometry = affinity.scale(geometry, *(canvas_size / patch_size), origin=(0, 0))
    try:
        intersection = geometry.intersection(canvas)
    except TopologicalError:
        geometry = geometry.buffer(0)
        intersection = geometry.intersection(canvas)

    if isinstance(intersection, Polygon):
        intersection = intersection.boundary
    shapes = (
        list(intersection.geoms)
        if isinstance(intersection, BaseMultipartGeometry)
        else [intersection]
    )
    coords_list = [
        shape.boundary.coords if isinstance(shape, Polygon) else shape.coords
        for shape in shapes
    ]
    coords_list = [
        np.array(coords).astype(np.int32) for coords in coords_list if len(coords) > 0
    ]
    return coords_list


def polygon_to_mask(
    mask: np.ndarray,
    polygon: np.ndarray,
    patch_corner: np.ndarray,
    patch_size: np.ndarray,
    canvas_size: np.ndarray,
):
    polygons = map_patch_to_canvas(
        Polygon(polygon[:, :2]), patch_corner, patch_size, canvas_size
    )
    if len(polygons) == 0:
        return mask
    return cv2.fillPoly(mask, polygons, 1)


def polyline_to_mask(
    mask: np.ndarray,
    polyline: np.ndarray,
    patch_corner: np.ndarray,
    patch_size: np.ndarray,
    canvas_size: np.ndarray,
):
    polylines = map_patch_to_canvas(
        LineString(polyline[:, :2]), patch_corner, patch_size, canvas_size
    )
    if len(polylines) == 0:
        return mask
    return cv2.polylines(mask, polylines, False, 1, 2)
