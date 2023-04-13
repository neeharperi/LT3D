import os
import pickle
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import numpy.typing as npt
from mmdet3d.datasets import AV2Dataset, NuScenesDataset
from tqdm import tqdm

NUSCENES_CLASS_NAMES = NuScenesDataset.CLASSES
AV2_CLASS_NAMES = AV2Dataset.CLASSES


NDArrayFloat = npt.NDArray[np.float64]
Frame = Dict[str, Any]
Frames = List[Frame]
Sequences = Dict[str, Frames]


def progressbar(itr: Iterable, desc: Optional[str] = None, **kwargs) -> Iterable:
    pbar = tqdm(itr, **kwargs)
    if desc:
        pbar.set_description(desc)
    return pbar


def save(obj, path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path: str) -> Any:
    """
    Returns
    -------
        object or None: returns None if the file does not exist
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def unpack_predictions(frames: Frames, classes: List[str]) -> Frames:
    """Convert data from mmdetection3D format to numpy format.

    Args:
        frames: list of frames
        classes: list of class names

    Returns:
        List of prediction item where each is a dictionary with keys:
            translation: ndarray[instance, [x, y, z]]
            size: ndarray[instance, [l, w, h]]
            yaw: ndarray[instance, float]
            velocity: ndarray[instance, [x, y]]
            label: ndarray[instance, int]
            score: ndarray[instance, float]
            frame_index: ndarray[instance, int]
    """
    unpacked_frames = []
    for frame_dict in frames:
        prediction = frame_dict["pts_bbox"]
        bboxes = prediction["boxes_3d"].tensor.numpy()
        label = prediction["labels_3d"].numpy()
        unpacked_frames.append(
            {
                "translation": bboxes[:, :3],
                "size": bboxes[:, 3:6],
                "yaw": wrap_pi(bboxes[:, 6]),
                "velocity": bboxes[:, -2:],
                "label": label,
                "name": np.array(
                    [classes[id] if id < len(classes) else "OTHER" for id in label]
                ),
                "score": prediction["scores_3d"].numpy(),
            }
        )
    return unpacked_frames


def annotate_frame_metadata(
    prediction_frames: Frames, label_frames: Frames, metadata_keys: List[str]
) -> None:
    """Copy annotations with provided keys from label to prediction frames.

    Args:
        prediction_frames: list of prediction frames
        label_frames: list of label frames
        metadata_keys: keys of the annotations to be copied
    """
    assert len(prediction_frames) == len(label_frames)
    for prediction, label in zip(prediction_frames, label_frames):
        for key in metadata_keys:
            prediction[key] = label[key]


def group_frames(frames_list: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Parameters
    ----------
    frames_list: list
        list of frames, each containing a detections snapshot for a timestamp
    """
    frames_by_seq_id = defaultdict(list)
    frames_list = sorted(frames_list, key=lambda f: f["timestamp_ns"])
    for frame in frames_list:
        frames_by_seq_id[frame["seq_id"]].append(frame)
    return dict(frames_by_seq_id)


def ungroup_frames(frames_by_seq_id: Dict[str, List[Dict]]):
    ungrouped_frames = []
    for frames in frames_by_seq_id.values():
        ungrouped_frames.extend(frames)
    return ungrouped_frames


def index_array_values(array_dict: Dict, index: Union[int, np.ndarray]) -> Dict:
    return {
        k: v[index] if isinstance(v, np.ndarray) else v for k, v in array_dict.items()
    }


def array_dict_iterator(array_dict: Dict, length: int):
    return (index_array_values(array_dict, i) for i in range(length))


def concatenate_array_values(array_dicts: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Concatenates numpy arrays in list of dictionaries
    Handles inconsistent keys (will skip missing keys)
    Does not concatenate non-numpy values (int, str), sets to value if all values are equal
    """
    combined = defaultdict(list)
    for array_dict in array_dicts:
        for k, v in array_dict.items():
            combined[k].append(v)
    concatenated = {}
    for k, vs in combined.items():
        if all(isinstance(v, np.ndarray) for v in vs):
            if any(v.size > 0 for v in vs):
                concatenated[k] = np.concatenate([v for v in vs if v.size > 0])
            else:
                concatenated[k] = vs[0]
        elif all(vs[0] == v for v in vs):
            concatenated[k] = vs[0]
    return concatenated


def filter_by_class_names(frames_by_seq_id: Dict, class_names) -> Dict:
    frames = ungroup_frames(frames_by_seq_id)
    return group_frames(
        [
            index_array_values(frame, np.isin(frame["name"], class_names))
            for frame in frames
        ]
    )


def filter_by_class_thresholds(
    frames_by_seq_id: Dict, thresholds_by_class: Dict[str, float]
) -> Dict:
    frames = ungroup_frames(frames_by_seq_id)
    return group_frames(
        [
            concatenate_array_values(
                [
                    index_array_values(
                        frame,
                        (frame["name"] == class_name) & (frame["score"] >= threshold),
                    )
                    for class_name, threshold in thresholds_by_class.items()
                ]
            )
            for frame in frames
        ]
    )


def filter_by_ego_xy_distance(frames_by_seq_id: Sequences, distance_threshold: float):
    frames = ungroup_frames(frames_by_seq_id)
    return group_frames(
        [
            index_array_values(
                frame,
                np.linalg.norm(
                    frame["translation"][:, :2]
                    - np.array(frame["ego_translation"])[:2],
                    axis=1,
                )
                <= distance_threshold,
            )
            for frame in frames
        ]
    )


def group_by_track_id(frames: Frames) -> Sequences:
    tracks_by_track_id = defaultdict(list)
    for frame_idx, frame in enumerate(frames):
        for instance in array_dict_iterator(frame, len(frame["translation"])):
            instance["frame_idx"] = frame_idx
            tracks_by_track_id[instance["track_id"]].append(instance)
    return dict(tracks_by_track_id)


def wrap_pi(theta: NDArrayFloat) -> NDArrayFloat:
    theta = np.remainder(theta, 2 * np.pi)
    theta[theta > np.pi] -= 2 * np.pi
    return theta
