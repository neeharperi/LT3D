"""
Paths:
dataset dir (val/train): contains gt information, city pose information, under user
downsampled dataset dir: generated by mmdet3d, under user
detector predictions dir (config: val/train): typically under mmdet3d

Args
max_age: int
    default 3
score_threshold: float
    default 0.1
"""

import os
from argparse import ArgumentParser

from data.utils import *
from pathlib import Path
from data.paths import PATHS
from tracking.ab3dmot import AB3DMOT
from tracking.greedy_tracker import CLS_VELOCITY_ERROR_BY_DATASET, GreedyTracker
from tracking.utils import average_scores_across_track
from pprint import pprint 

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="av2", choices=["av2", "nuscenes"])
    argparser.add_argument("--split", default="val", choices=["val", "test"])
    argparser.add_argument("--tracker", default="greedy_tracker", choices=["greedy_tracker", "ab3dmot_tracker"])
    argparser.add_argument("--objective_metric", default="HOTA", choices=["HOTA", "MOTA"])
    
    argparser.add_argument("--max_track_age", type=int, default=3)
    argparser.add_argument("--score_threshold", type=float, default=0.1)
    argparser.add_argument("--keep_top_k", type=int, default=500)
    argparser.add_argument("--num_score_thresholds", type=int, default=10)
    argparser.add_argument("--ego_distance_threshold", type=int, default=50)
    config = argparser.parse_args()

    # load data
    paths = PATHS[config.dataset][config.split]
    if config.dataset == "av2":
        from data.av2_adapter import av2_load_predictions_and_labels

        classes = AV2_CLASS_NAMES
        predictions, labels = av2_load_predictions_and_labels(
            paths["prediction"], paths["infos"], paths["dataset_dir"]
        )
    elif config.dataset == "nuscenes":
        from nuscenes import NuScenes
        from data.nuscenes_adapter import nuscenes_load_predictions_and_labels

        nusc_version = "v1.0-trainval"
        nusc = NuScenes(nusc_version, dataroot=paths["dataset_dir"])
        classes = NUSCENES_CLASS_NAMES
        predictions, labels = nuscenes_load_predictions_and_labels(
            paths["prediction"], paths["infos"], nusc
        )

    # generate tracks
    if config.tracker == "greedy_tracker":
        tracker = GreedyTracker(
            CLS_VELOCITY_ERROR_BY_DATASET[config.dataset], max_age=config.max_track_age
        )
    elif config.tracker == "ab3dmot_tracker":
        tracker = AB3DMOT(classes=classes, max_age=config.max_track_age)
    else:
        raise Exception(f"Tracker {config.tracker} not found")
    track_predictions = {}
    for seq_id, frames in progressbar(predictions.items(), "running tracker on logs"):
        tracker.reset()
        last_t = frames[0]["timestamp_ns"]
        track_predictions[seq_id] = []
        for frame in frames:
            time_delta_seconds = (frame["timestamp_ns"] - last_t) * 1e-9
            last_t = frame["timestamp_ns"]
            # filter predictions with confidence below threshold
            filtered_predictions = index_array_values(
                frame, frame["score"] > config.score_threshold
            )
            # only keep top K predictions
            k = min(len(filtered_predictions["score"]), config.keep_top_k) - 1
            filtered_predictions = index_array_values(
                filtered_predictions,
                filtered_predictions["score"]
                >= -np.partition(-filtered_predictions["score"], k)[k],
            )
            track_frame = tracker.step(filtered_predictions, time_delta_seconds)
            # only store active tracks
            track_frame = index_array_values(track_frame, track_frame["active"] > 0)
            for k, v in frame.items():
                if not isinstance(v, np.ndarray):
                    track_frame[k] = v
            track_predictions[seq_id].append(track_frame)

    # average scores across a single track
    track_predictions = average_scores_across_track(track_predictions)

    # save everything
    outputs_dir = os.path.join(
        "results", f"{config.dataset}-{config.split}", config.tracker, "outputs"
    )
    print(f"Saving track predictions to {outputs_dir}")
    save(labels, os.path.join("dataset", f"{config.dataset}-{config.split}", "labels.pkl"))
    save(predictions, os.path.join(outputs_dir, "detection_predictions.pkl"))
    save(track_predictions, os.path.join(outputs_dir, "track_predictions.pkl"))
    
    if config.split != "test":
        if config.dataset == "av2":
            from av2.evaluation.tracking.eval import evaluate
            res =  evaluate(track_predictions, labels, config.objective_metric, config.ego_distance_threshold, paths["dataset_dir"], outputs_dir)
            pprint(res)
        elif config.dataset == "nuscenes":
            raise Exception(f"Not Implemented Yet")
                
        save(res, os.path.join(outputs_dir, "res.pkl"))
