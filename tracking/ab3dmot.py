"""
Adapted from https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
"""
from __future__ import print_function

import copy
from typing import Dict, List

import numpy as np
from filterpy.kalman import KalmanFilter
from numba import jit
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

from data.utils import concatenate_array_values, index_array_values


@jit
def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@jit
def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


@jit
def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    """Clip a polygon with another polygon.
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def iou3d(corners1, corners2):
    """Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    """
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


@jit
def roty(t: float):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@jit
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def convert_3dbox_to_8corner(bbox3d_input):
    """Takes an object and a projection matrix (P) and projects the 3d
    bounding box into the image plane.
    Returns:
        corners_2d: (8,2) array in left image coord.
        corners_3d: (8,3) array in in rect camera coord.
    Note: the output of this function will be passed to the funciton iou3d
        for calculating the 3D-IOU. But the function iou3d was written for
        kitti
    """
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    # transform to kitti format first
    bbox3d_nuscenes = copy.copy(bbox3d)
    # kitti:    [x,  y,  z,  a, l, w, h]
    # nuscenes: [y, -z, -x, -a, w, l, h]
    bbox3d[0] = bbox3d_nuscenes[1]
    bbox3d[1] = -bbox3d_nuscenes[2]
    bbox3d[2] = -bbox3d_nuscenes[0]
    bbox3d[3] = -bbox3d_nuscenes[3]
    bbox3d[4] = bbox3d_nuscenes[5]
    bbox3d[5] = bbox3d_nuscenes[4]

    R = roty(bbox3d[3])

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(
        self,
        bbox3D,
    ):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

        # Initialize the covariance matrix, see covariance.py for more details
        self.kf.P[
            7:, 7:
        ] *= 1000.0  # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.0
        self.kf.Q[7:, 7:] *= 0.01

        self.kf.x[:7] = bbox3D.reshape((7, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.last_observation = bbox3D

    def update(self, bbox3D):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

        ######################### orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if (
            abs(new_theta - predicted_theta) > np.pi / 2.0
            and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0
        ):  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi:
                self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        #########################

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        self.last_observation = bbox3D

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7,))


def angle_in_range(angle):
    """
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    """
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle


def diff_orientation_correction(det, trk):
    """
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    """
    diff = det - trk
    diff = angle_in_range(diff)
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    diff = angle_in_range(diff)
    return diff


def greedy_match(distance_matrix):
    """
    Find the one-to-one matching using greedy allgorithm choosing small distance
    distance_matrix: (num_detections, num_tracks)
    """
    matched_indices = []

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if (
            tracking_id_matches_to_detection_id[tracking_id] == -1
            and detection_id_matches_to_tracking_id[detection_id] == -1
        ):
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)
    return matched_indices


def associate_detections_to_tracks(
    detections,
    tracks,
    iou_threshold=0.1,
    match_algorithm="greedy",
):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    detections:  N x 8 x 3
    tracks:    M x 8 x 3

    dets: N x 7
    trks: M x 7
    trks_S: N x 7 x 7

    Returns 3 lists of matches, unmatched_detections and unmatched_tracks
    """
    if len(tracks) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 8, 3), dtype=int),
        )
    iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
    distance_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(tracks):
            iou_matrix[d, t] = iou3d(det, trk)[0]  # det: 8 x 3, trk: 8 x 3
    distance_matrix = -iou_matrix

    if match_algorithm == "greedy":
        matched_indices = greedy_match(distance_matrix)
    elif match_algorithm == "hungarian":
        matched_indices = linear_sum_assignment(distance_matrix)  # hungarian algorithm
        matched_indices = np.stack(matched_indices, axis=-1)
    else:
        raise ValueError()

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_tracks = []
    for t, trk in enumerate(tracks):
        if len(matched_indices) == 0 or (t not in matched_indices[:, 1]):
            unmatched_tracks.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        match = True
        if iou_matrix[m[0], m[1]] < iou_threshold:
            match = False
        if not match:
            unmatched_detections.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


class SingleClassMOT(object):
    def __init__(
        self,
        match_threshold: float,
        match_algorithm: str,
        max_age=2,
    ):
        """
        observation:
          before reorder: [h, w, l, x, y, z, rot_y]
          after reorder:  [x, y, z, rot_y, l, w, h]
        state:
          [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
        """
        self.match_threshold = match_threshold
        self.match_algorithm = match_algorithm
        self.max_age = max_age
        self.tracks: List[KalmanBoxTracker] = []
        self.track_id_to_detection = {}

    def update(self, detections):
        """
        Requires: this method must be called once for each frame even with empty detections.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # new change: keep everything in the format [x, y, z, yaw, l, w, h]
        # self.reorder = [3, 4, 5, 6, 2, 1, 0]
        # self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        # # before reorder the format is [h, w, l, x, y, z, yaw], the original function docstring was wrong to say it is [x, y, z, yaw, l, w, h]
        # detections = detections[:, self.reorder]
        # # assumes det format is [x, y, z, yaw, l, w, h]
        # # after reordering, if we assume the state is in the same format as detections
        # # [h, w, l, x, y, z, yaw] -> correct

        bboxes = np.concatenate(
            [
                detections["translation"],
                detections["yaw"].reshape(-1, 1),
                detections["size"],
            ],
            axis=-1,
        )

        track_predictions = [track.predict().reshape(-1)[:7] for track in self.tracks]
        track_predictions = (
            np.stack(track_predictions, axis=0)
            if track_predictions
            else np.zeros((0, 7))
        )
        no_nans = np.all(~np.isnan(track_predictions), axis=1)
        track_predictions = track_predictions[no_nans]
        self.tracks = [track for not_nan, track in zip(no_nans, self.tracks) if not_nan]

        detections_8corner = [convert_3dbox_to_8corner(bbox) for bbox in bboxes]
        if len(detections_8corner) > 0:
            detections_8corner = np.stack(detections_8corner, axis=0)

        track_predictions_8corner = [
            convert_3dbox_to_8corner(track_prediction)
            for track_prediction in track_predictions
        ]
        if len(track_predictions_8corner) > 0:
            track_predictions_8corner = np.stack(track_predictions_8corner, axis=0)

        (
            matched,
            unmatched_detections,
            unmatched_track_predictions,
        ) = associate_detections_to_tracks(
            detections_8corner,
            track_predictions_8corner,
            iou_threshold=self.match_threshold,
            match_algorithm=self.match_algorithm,
        )

        # update matched trackers with assigned detections
        for idx, track in enumerate(self.tracks):
            if idx not in unmatched_track_predictions:
                detection_idx = matched[np.where(matched[:, 1] == idx)[0][0], 0]
                track.update(bboxes[detection_idx, :])
                self.track_id_to_detection[track.id] = index_array_values(
                    detections, np.array([detection_idx])
                )

        # create and initialise new trackers for unmatched detections
        for idx in unmatched_detections:  # a scalar of index
            track = KalmanBoxTracker(
                bboxes[idx, :],
            )
            self.tracks.append(track)
            self.track_id_to_detection[track.id] = index_array_values(
                detections, np.array([idx])
            )

        self.tracks = [
            track for track in self.tracks if track.time_since_update < self.max_age
        ]
        return [
            {
                **self.track_id_to_detection[track.id],
                "track_id": np.array([track.id]),
                "active": np.array([int(track.time_since_update == 0)]),
            }
            for track in self.tracks
        ]


class AB3DMOT:
    def __init__(self, classes: List[str], max_age: int = 2):
        self.classes = classes
        self.max_age = max_age
        self.reset()

    def reset(self):
        self.trackers = {
            name: SingleClassMOT(
                match_threshold=0.1,
                match_algorithm="hungarian",
                max_age=self.max_age,
            )
            for name in self.classes
        }

    def step(self, detections: Dict, time_delta: float):
        """
        reorders input to support consistent tracker API
        """
        tracks = []
        for name, tracker in self.trackers.items():
            detections_in_class = index_array_values(
                detections, detections["name"] == name
            )
            tracks.extend(tracker.update(detections_in_class))

        return concatenate_array_values(tracks)
