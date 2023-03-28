"""Argoverse Tracking evaluation

Evaluation Metrics:
    HOTA: see https://arxiv.org/abs/2009.07736
    MOTA: see https://jivp-eurasipjournals.springeropen.com/articles/10.1155/2008/246309
    AMOTA: see https://arxiv.org/abs/2008.08063
"""
import contextlib
from copy import copy
from functools import partial
from itertools import chain
from typing import Callable, Dict, Final, List, Tuple, Union

import numpy as np
import trackeval
from scipy.optimize import linear_sum_assignment
from trackeval.datasets._base_dataset import _BaseDataset

from data.utils import *

SUBMETRIC_TO_METRIC_CLASS_NAME: Final[Dict[str, str]] = {
    "MOTA": "CLEAR",
    "HOTA": "HOTA",
}


class TrackEvalDataset(_BaseDataset):
    def __init__(self, config: Dict):
        super().__init__()
        self.gt_tracks = config["GT_TRACKS"]
        self.predicted_tracks = config["PREDICTED_TRACKS"]
        self.full_class_list = config.get("CLASSES", config["CLASSES_TO_EVAL"])
        self.class_list = config["CLASSES_TO_EVAL"]
        self.tracker_list = config["TRACKERS_TO_EVAL"]
        self.seq_list = config["SEQ_IDS_TO_EVAL"]
        self.output_fol = config["OUTPUT_FOLDER"]
        self.output_sub_fol = config["OUTPUT_SUB_FOLDER"]
        self.zero_distance = config["ZERO_DISTANCE"]
        print(f"Using zero_distance={self.zero_distance}m")

    @staticmethod
    def get_default_dataset_config() -> Dict:
        default_config = {
            "GT_TRACKS": None,  # tracker_name -> seq id -> frames
            "PREDICTED_TRACKS": None,  # tracker_name -> seq id -> frames
            "SEQ_IDS_TO_EVAL": None,  # list of sequence ids to eval
            "CLASSES_TO_EVAL": None,
            "TRACKERS_TO_EVAL": None,
            "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            "OUTPUT_SUB_FOLDER": "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            "ZERO_DISTANCE": 2,
        }
        return default_config

    def _load_raw_file(
        self, tracker: str, seq_id: Union[str, int], is_gt: bool
    ) -> Dict[str, Union[List[np.ndarray], float]]:
        """
        get raw track data, from either trackers or ground truth
        """
        tracks = (self.gt_tracks if is_gt else self.predicted_tracks)[tracker][seq_id]
        source = "gt" if is_gt else "tracker"

        ts = np.array([frame["timestamp_ns"] for frame in tracks])
        assert np.all(ts[:-1] < ts[1:]), "timestamps are not increasing"

        raw_data = {
            f"{source}_ids": [frame["track_id"] for frame in tracks],
            f"{source}_classes": [frame["label"] for frame in tracks],
            f"{source}_dets": [
                np.concatenate((frame["translation"], frame["size"]), axis=-1)
                for frame in tracks
            ],
            "num_timesteps": len(tracks),
            "seq": seq_id,
        }
        if "score" in tracks[0]:
            raw_data[f"{source}_confidences"] = [frame["score"] for frame in tracks]
        return raw_data

    def get_preprocessed_seq_data(
        self, raw_data: Dict[str, Union[List[np.ndarray], float]], cls: str
    ) -> Dict[str, Union[List[np.ndarray], float]]:
        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_classes",
            "tracker_classes",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
            "num_timesteps",
            "seq",
        ]
        data = {k: copy(raw_data[k]) for k in data_keys}
        cls_id = self.full_class_list.index(cls)

        for t in range(raw_data["num_timesteps"]):
            gt_to_keep_mask = data["gt_classes"][t] == cls_id
            data["gt_classes"][t] = data["gt_classes"][t][gt_to_keep_mask]
            data["gt_ids"][t] = data["gt_ids"][t][gt_to_keep_mask]
            data["gt_dets"][t] = data["gt_dets"][t][gt_to_keep_mask, :]

            tracker_to_keep_mask = data["tracker_classes"][t] == cls_id
            data["tracker_classes"][t] = data["tracker_classes"][t][
                tracker_to_keep_mask
            ]
            data["tracker_ids"][t] = data["tracker_ids"][t][tracker_to_keep_mask]
            data["tracker_dets"][t] = data["tracker_dets"][t][tracker_to_keep_mask, :]
            data["tracker_confidences"][t] = data["tracker_confidences"][t][
                tracker_to_keep_mask
            ]

            data["similarity_scores"][t] = data["similarity_scores"][t][
                :, tracker_to_keep_mask
            ][gt_to_keep_mask]

        # map ids to 0 - n
        unique_gt_ids = set(chain.from_iterable(data["gt_ids"]))
        unique_tracker_ids = set(chain.from_iterable(data["tracker_ids"]))
        data["gt_ids"] = self._map_ids(data["gt_ids"], unique_gt_ids)
        data["tracker_ids"] = self._map_ids(data["tracker_ids"], unique_tracker_ids)

        data["num_tracker_dets"] = sum(len(dets) for dets in data["tracker_dets"])
        data["num_gt_dets"] = sum(len(dets) for dets in data["gt_dets"])
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _map_ids(self, ids: List[np.ndarray], unique_ids: List) -> List[np.ndarray]:
        id_map = {id: i for i, id in enumerate(unique_ids)}
        return [
            np.array([id_map[id] for id in id_array], dtype=int) for id_array in ids
        ]

    def _calculate_similarities(
        self, gt_dets_t: np.ndarray, tracker_dets_t: np.ndarray
    ) -> np.ndarray:
        """
        Euclidean distance of the x, y translation coordinates
        """
        gt_xy = gt_dets_t[:, :2]
        tracker_xy = tracker_dets_t[:, :2]
        return self._calculate_euclidean_similarity(
            gt_xy, tracker_xy, zero_distance=self.zero_distance
        )


def evaluate(
    labels: Dict,
    track_predictions: Dict,
    classes: List[str],
    tracker_name: str,
    output_dir: str,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Evaluates a set of tracks against ground truth annotations using the TrackEval evaluation suite.
    Each sequence/log is evaluated separately.

    Args:
        labels: Dict[seq_id: List[frame]] Dictionary of ground truth annotations
        track_predictions: Dict[seq_id: List[frame]] Dictionary of tracks
        classes: List of classes to evaluate
        tracker_name: Name of tracker
        output_dir: Folder to save evaluation results
        iou_threshold: IoU threshold for a True Positive match betwee a detection to a ground truth bounding box

    frame is a dictionary with the following format
        {
            sequence_id: [
                {
                    "timestamp_ns": int, # nano seconds
                    "track_id": np.ndarray[I],
                    "translation": np.ndarray[I, 3],
                    "size": np.ndarray[I, 3],
                    "yaw": np.ndarray[I],
                    "velocity": np.ndarray[I, 3],
                    "label": np.ndarray[I],
                    "score": np.ndarray[I],
                    "name": np.ndarray[I],
                    ...
                }
            ]
        }
    where I is the number of objects in the frame

    Returns:
        dictionary of metric values
    """
    labels_id_ts = set(
        (frame["seq_id"], frame["timestamp_ns"]) for frame in ungroup_frames(labels)
    )
    predictions_id_ts = set(
        (frame["seq_id"], frame["timestamp_ns"])
        for frame in ungroup_frames(track_predictions)
    )
    assert (
        labels_id_ts == predictions_id_ts
    ), "sequence ids and timestamp_ns in labels and predictions don't match"
    metrics_config = {
        "METRICS": ["HOTA", "CLEAR"],
        "THRESHOLD": iou_threshold,
    }
    metrics_list = [
        getattr(trackeval.metrics, metric)(metrics_config)
        for metric in metrics_config["METRICS"]
    ]
    dataset_config = {
        **TrackEvalDataset.get_default_dataset_config(),
        "GT_TRACKS": {tracker_name: labels},
        "PREDICTED_TRACKS": {tracker_name: track_predictions},
        "SEQ_IDS_TO_EVAL": list(labels.keys()),
        "CLASSES_TO_EVAL": classes,
        "TRACKERS_TO_EVAL": [tracker_name],
        "OUTPUT_FOLDER": output_dir,
    }

    evaluator = trackeval.Evaluator(
        {
            **trackeval.Evaluator.get_default_eval_config(),
            "TIME_PROGRESS": False,
        }
    )
    full_result, eval_msg = evaluator.evaluate(
        [TrackEvalDataset(dataset_config)],
        metrics_list,
    )

    return full_result


def tune_score_thresholds(
    labels: Dict,
    track_predictions: Dict,
    objective_metric: str,
    classes: List[str],
    num_thresholds: int = 10,
    iou_threshold: float = 0.5,
    match_distance_threshold: int = 2,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Finds the optimal score thresholds to optimize the objective metric.
    Each class is processed independently.

    Args:
        labels: Dict[seq_id: List[frame]] Dictionary of ground truth annotations
        track_predictions: Dict[seq_id: List[frame]] Dictionary of tracks
        objective_metric: Name of the metric to optimize, one of HOTA or MOTA
        classes: List of classes to evaluate
        iou_threshold: IoU threshold for a True Positive match betwee a detection to a ground truth bounding box

    Returns:
        optimal_score_threshold_by_class: Dictionary of class name to optimal score threshold
        optimal_metric_values_by_class: Dictionary of class name to metric value with the optimal score threshold
        mean_metric_values_by_class: Dictionary of class name to metric value averaged over recall levels
    """
    metric_class = SUBMETRIC_TO_METRIC_CLASS_NAME[objective_metric]
    metrics_config = {
        "METRICS": [metric_class],
        "THRESHOLD": iou_threshold,
        "PRINT_CONFIG": False,
    }
    metrics_list = [
        getattr(trackeval.metrics, metric_name)(metrics_config)
        for metric_name in metrics_config["METRICS"]
    ]
    dataset_config = {
        **TrackEvalDataset.get_default_dataset_config(),
        "GT_TRACKS": {"tracker": labels},
        "PREDICTED_TRACKS": {"tracker": track_predictions},
        "SEQ_IDS_TO_EVAL": list(labels.keys()),
        "CLASSES_TO_EVAL": classes,
        "TRACKERS_TO_EVAL": ["tracker"],
        "OUTPUT_FOLDER": "tmp",
    }
    evaluator = trackeval.Evaluator(
        {
            **trackeval.Evaluator.get_default_eval_config(),
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
    )

    score_thresholds_by_class = {}
    sim_func = partial(xy_center_similarity, zero_distance=match_distance_threshold)
    for name in classes:
        filter_by_class = lambda detections: group_frames(
            [
                index_array_values(f, f["name"] == name)
                for f in ungroup_frames(detections)
            ]
        )
        single_cls_labels = filter_by_class(labels)
        single_cls_predictions = filter_by_class(track_predictions)
        score_thresholds_by_class[name] = calculate_score_thresholds(
            single_cls_labels,
            single_cls_predictions,
            sim_func,
            num_thresholds=num_thresholds,
        )

    metric_results = []
    for threshold_i in progressbar(
        range(num_thresholds), "calculating optimal track score thresholds"
    ):
        score_threshold_by_class = {
            n: score_thresholds_by_class[n][threshold_i] for n in classes
        }
        filtered_predictions = filter_by_class_thresholds(
            track_predictions, score_threshold_by_class
        )
        with contextlib.redirect_stdout(
            None
        ):  # silence print statements from TrackEval
            result_for_threshold, _ = evaluator.evaluate(
                [
                    TrackEvalDataset(
                        {
                            **dataset_config,
                            "PREDICTED_TRACKS": {"tracker": filtered_predictions},
                        }
                    )
                ],
                metrics_list,
            )
        metric_results.append(
            result_for_threshold["TrackEvalDataset"]["tracker"]["COMBINED_SEQ"]
        )

    optimal_score_threshold_by_class = {}
    optimal_metric_values_by_class = {}
    mean_metric_values_by_class = {}
    for name in classes:
        metric_values = [
            r[name][metric_class][objective_metric] for r in metric_results
        ]
        metric_values = [
            np.mean(v) if isinstance(v, np.ndarray) else v for v in metric_values
        ]
        optimal_threshold = score_thresholds_by_class[name][np.argmax(metric_values)]
        optimal_score_threshold_by_class[name] = optimal_threshold
        optimal_metric_values_by_class[name] = max(0, np.max(metric_values))
        mean_metric_values_by_class[name] = np.nanmean(
            np.array(metric_values).clip(min=0)
        )
    return (
        optimal_score_threshold_by_class,
        optimal_metric_values_by_class,
        mean_metric_values_by_class,
    )


def calculate_score_thresholds(
    labels: Dict,
    predictions: Dict,
    sim_func: Callable,
    num_thresholds: int = 40,
    min_recall: float = 0.1,
) -> np.ndarray:
    scores, n_gt = calculate_matched_scores(labels, predictions, sim_func)
    recall_thresholds = np.linspace(min_recall, 1, num_thresholds).round(12)[::-1]
    if len(scores) == 0:
        return np.zeros_like(recall_thresholds)
    score_thresholds = recall_to_scores(
        scores, recall_threshold=recall_thresholds, n_gt=n_gt
    )
    score_thresholds = np.nan_to_num(score_thresholds, nan=0)
    return score_thresholds


def calculate_matched_scores(
    labels: Dict, predictions: Dict, sim_func: Callable
) -> Tuple[np.ndarray, int]:
    scores = []
    n_gt = 0
    num_tp = 0
    for seq_id in labels:
        for label_frame, prediction_frame in zip(labels[seq_id], predictions[seq_id]):
            sim = sim_func(label_frame["translation"], prediction_frame["translation"])
            match_rows, match_cols = linear_sum_assignment(-sim)
            scores.append(prediction_frame["score"][match_cols])
            n_gt += len(label_frame["translation"])
            num_tp += len(match_cols)

    scores = np.concatenate(scores)
    return scores, n_gt


def recall_to_scores(
    scores: np.ndarray, recall_threshold: np.ndarray, n_gt: int
) -> np.ndarray:
    # Sort scores.
    scores.sort()
    scores = scores[::-1]

    # Determine thresholds.
    recall_values = np.arange(1, len(scores) + 1) / n_gt
    max_recall_achieved = np.max(recall_values)
    assert max_recall_achieved <= 1
    score_thresholds = np.interp(recall_threshold, recall_values, scores, right=0)

    # Set thresholds for unachieved recall values to nan to penalize AMOTA/AMOTP later.
    if isinstance(recall_threshold, np.ndarray):
        score_thresholds[recall_threshold > max_recall_achieved] = np.nan
    return score_thresholds


def xy_center_similarity(
    centers1: np.ndarray, centers2: np.ndarray, zero_distance: float
) -> np.ndarray:
    if centers1.size == 0 or centers2.size == 0:
        return np.zeros((len(centers1), len(centers2)))
    xy_dist = np.linalg.norm(
        centers1[:, np.newaxis, :2] - centers2[np.newaxis, :, :2], axis=2
    )
    sim = np.maximum(0, 1 - xy_dist / zero_distance)
    return sim
