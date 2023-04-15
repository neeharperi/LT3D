from typing import Dict

import numpy as np

from data.utils import *


def forecast(
    tracks: Dict,
    velocity_scalar: list,
    num_timesteps: int = 6,
    time_delta: float = 0.5,
):
    velocity_scalar = np.array(velocity_scalar)

    linear_forecasts = {}
    for seq_id in progressbar(tracks.keys(), desc="generating linear forecasts"):
        linear_forecasts[seq_id] = {}
        for frame in tracks[seq_id]:
            timestamp_ns = frame["timestamp_ns"]
            linear_forecasts[seq_id][timestamp_ns] = []
            for agent in array_dict_iterator(frame, len(frame["track_id"])):
                multiplier = (
                    velocity_scalar[:, np.newaxis, np.newaxis]
                    * np.arange(num_timesteps)[np.newaxis, :, np.newaxis]
                    * time_delta
                )
                prediction = (
                    agent["translation_m"][:2]
                    + multiplier * agent["velocity"][np.newaxis, np.newaxis, :2]
                )
                linear_forecasts[seq_id][timestamp_ns].append(
                    {
                        "timestep_ns": timestamp_ns,
                        "current_translation_m": agent["translation_m"][:2],
                        "detection_score": agent["score"],
                        "size": agent["size"],
                        "label": agent["label"],
                        "name": agent["name"],
                        "yaw": agent["yaw"],
                        "prediction_m": prediction,
                        "score": np.stack(5 * [agent["score"]]),
                        "instance_id": agent["track_id"],
                    }
                )

    return linear_forecasts
