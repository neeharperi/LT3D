"""
Adapted from https://github.com/tianweiy/CenterPoint
"""
from copy import deepcopy
from typing import Dict

import numpy as np

from data.utils import concatenate_array_values, index_array_values

INVALID_DIST = 1e18


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = INVALID_DIST
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)


class GreedyTracker(object):
    def __init__(self, cls_velocity_error, max_age=0):
        self.cls_velocity_error = cls_velocity_error
        self.max_age = max_age
        self.reset()

    def reset(self):
        self.id_count = 0
        self.tracks: Dict[str, np.ndarray] = {
            k: np.array([])
            for k in [
                "translation",
                "xy",
                "xy_velocity",
                "label",
                "track_id",
                "active",
                "age",
            ]
        }

    def step(self, detections: Dict, time_lag: float) -> Dict[str, np.ndarray]:
        """
        Parameters
        ----------
        detections: dict
            contains the keys:
            translation: ndarray[I, 3]
            size: ndarray[I, 3]
            yaw: ndarray[I]
            velocity: ndarray[I, 2]
            label: ndarray[I]
            score: ndarray[I]
            timestamp_ns: ndarray[I]
            name: ndarray[I]
            seq_id: str
        time_lag: float
            seconds
        Returns
        -------
        list
            list of tracks, each of which is a dictionary with keys
            label: int
            detection_name: str
            translation: ndarray
            velocity: ndarray
            track_id: int
            age: int, steps since first detection
            active: int, number of recent consecutive matched detections
        """
        detections = index_array_values(
            detections,
            np.array(
                [n in self.cls_velocity_error for n in detections["name"]], dtype=bool
            ),
        )
        N = len(detections["translation"])
        M = len(self.tracks["translation"])

        detections["xy"] = detections["translation"][:, :2]
        detections["xy_velocity"] = detections["velocity"][:, :2]
        max_diff = np.array(
            [self.cls_velocity_error[n] for n in detections["name"]], np.float32
        )
        track_label = self.tracks["label"]  # M
        track_last_xy = self.tracks["xy"]  # M x 2
        detection_label = detections["label"]
        detection_last_xy = detections["xy"] - time_lag * detections["xy_velocity"]
        dist = track_last_xy.reshape(1, -1, 2) - detection_last_xy.reshape(-1, 1, 2)
        dist = np.sqrt((dist**2).sum(axis=2))  # N x M
        invalid = (dist > max_diff.reshape(N, 1)) | (
            detection_label.reshape(N, 1) != track_label.reshape(1, M)
        )
        dist = dist + invalid * INVALID_DIST

        matched_indices = greedy_assignment(deepcopy(dist))
        unmatched_detection_indices = np.delete(np.arange(N), matched_indices[:, 0])
        unmatched_track_indices = np.delete(np.arange(M), matched_indices[:, 1])

        matched_detections = index_array_values(detections, matched_indices[:, 0])
        matched_detections["track_id"] = self.tracks["track_id"][matched_indices[:, 1]]
        matched_detections["active"] = self.tracks["active"][matched_indices[:, 1]] + 1
        matched_detections["age"] = np.ones(len(matched_indices), int)

        M_new = len(unmatched_detection_indices)
        unmatched_detections = index_array_values(
            detections, unmatched_detection_indices
        )
        unmatched_detections["track_id"] = np.arange(
            self.id_count, self.id_count + M_new
        )
        self.id_count += M_new
        unmatched_detections["active"] = np.ones(M_new, int)
        unmatched_detections["age"] = np.ones(M_new, int)

        # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output
        # the object in current frame
        unmatched_tracks = index_array_values(self.tracks, unmatched_track_indices)
        unmatched_tracks = index_array_values(
            unmatched_tracks, unmatched_tracks["age"] < self.max_age
        )
        unmatched_tracks["age"] += 1
        unmatched_tracks["active"] = np.zeros(len(unmatched_tracks["active"]), int)
        unmatched_tracks["xy"] += time_lag * unmatched_tracks["xy_velocity"]

        self.tracks = concatenate_array_values(
            [matched_detections, unmatched_detections, unmatched_tracks]
        )
        return self.tracks


# 99.9 percentile of the l2 velocity error distribution
# Tune this for your model should provide some improvement
AV2_CLS_VELOCITY_ERROR = {
    "REGULAR_VEHICLE": 16,
    "PEDESTRIAN": 2,
    "BICYCLIST": 10,
    "MOTORCYCLIST": 15.5,
    "WHEELED_RIDER": 6.5,
    "BOLLARD": 0.1,
    "CONSTRUCTION_CONE": 0.1,
    "SIGN": 0.25,
    "CONSTRUCTION_BARREL": 0.15,
    "STOP_SIGN": 2,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": 0.08,
    "LARGE_VEHICLE": 14.5,
    "BUS": 14,
    "BOX_TRUCK": 15,
    "TRUCK": 14,
    "VEHICULAR_TRAILER": 16,
    "TRUCK_CAB": 15,
    "SCHOOL_BUS": 18,
    "ARTICULATED_BUS": 14,
    "MESSAGE_BOARD_TRAILER": 0.75,
    "BICYCLE": 8.75,
    "MOTORCYCLE": 13,
    "WHEELED_DEVICE": 5.8,
    "WHEELCHAIR": 13,
    "STROLLER": 2,
    "DOG": 2,
}

NUSCENES_CLS_VELOCITY_ERROR = {
    "car": 4,
    "truck": 4,
    "bus": 5.5,
    "trailer": 3,
    "pedestrian": 1,
    "motorcycle": 13,
    "bicycle": 3,
    "construction_vehicle": 5,
    "emergency_vehicle": 5,
    "adult": 5,
    "child": 5,
    "police_officer": 5,
    "construction_worker": 5,
    "stroller": 5,
    "personal_mobility": 5,
    "pushable_pullable": 5,
    "debris": 5,
    "traffic_cone": 5,
    "barrier": 5,
}

CLS_VELOCITY_ERROR_BY_DATASET = {
    "av2": AV2_CLS_VELOCITY_ERROR,
    "nuscenes": NUSCENES_CLS_VELOCITY_ERROR,
}
