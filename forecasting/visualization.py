"""
AV2 Scene Visualization with Forecasts
"""
import io
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from PIL import Image
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation
from av2.evaluation.forecasting.utils import trajectory_type
from av2.evaluation.forecasting.constants import CATEGORY_TO_VELOCITY_M_PER_S

from av2.evaluation.detection.utils import load_mapped_avm_and_egoposes
from av2.evaluation.forecasting.eval import convert_forecast_labels
import av2.rendering.vector as vector_plotting_utils
from data.utils import *
from data.paths import PATHS


def get_rectangle(cx, cy, w, h, rotation, **kwargs):
    x = cx - w/2 * np.cos(rotation) + h/2 * np.sin(rotation)
    y = cy - w/2 * np.sin(rotation) - h/2 * np.cos(rotation)
    return Rectangle((x, y), w, h, rotation / np.pi * 180, **kwargs)

def color_lookup(name):
    if name in ("REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "TRUCK_CAB", "SCHOOL_BUS", "ARTICULATED_BUS",):
        return "green"
    elif name == "PEDESTRIAN":
        return "blue"
    elif name in ("BICYCLIST", "MOTORCYCLIST", "WHEELED_RIDER", "BICYCLE", "MOTORCYCLE",):
        return "red"
    elif name == "LABEL":
        return "gray"
    elif name == "EGO":
        return "gold"
    return "black"


def make_animation(
    seq_id, 
    forecasts,
    labels,
    dataset_dir,
    only_show_most_likely=False,
    ego_distance_limit=50,
    only_show_detection_scores_above=0.4,
    skip_static_predictions=True,
):
    log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes([seq_id], Path(dataset_dir))
    avm = log_id_to_avm[seq_id]
    images = []
    min_timestamp, max_timestamp = min(forecasts[seq_id].keys()), max(forecasts[seq_id].keys())
    for timestamp_ns in sorted(forecasts[seq_id].keys())[:-1]:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        # add timestamp title
        plt.title(f"{(timestamp_ns - min_timestamp) / 1e9:.2f} / {(max_timestamp - min_timestamp) / 1e9:.2f}s")
        for _, ls in avm.vector_lane_segments.items():
            vector_plotting_utils.draw_polygon_mpl(ax, ls.polygon_boundary, color="grey", linewidth=0.5)
    
        # calculate center
        center = log_id_to_timestamped_poses[seq_id][timestamp_ns].translation[:2]
        ego_yaw = Rotation.from_matrix(log_id_to_timestamped_poses[seq_id][timestamp_ns].rotation).as_euler("zyx")[0]
        plt.xlim(center[0] - ego_distance_limit, center[0] + ego_distance_limit)
        plt.ylim(center[1] - ego_distance_limit, center[1] + ego_distance_limit)
        plt.axis('off')
        plt.scatter(*center, c="gold", label="ego")
        plt.gca().add_patch(get_rectangle(*center, 4.5, 2, ego_yaw, fill=False, color=color_lookup("EGO")))

        label_forecasts = labels[seq_id].get(timestamp_ns, [])
        for agent_forecast in label_forecasts + forecasts[seq_id][timestamp_ns]:
            # skip static classes
            if agent_forecast["name"] in ("BOLLARD", "CONSTRUCTION_CONE", "SIGN", "CONSTRUCTION_BARREL", "STOP_SIGN", "MOBILE_PEDESTRIAN_CROSSING_SIGN",):
                continue
            # skip objects further than d away
            xy = agent_forecast["current_translation_m"]
            if np.linalg.norm(xy - center) > ego_distance_limit:
                continue
            # check if it is a label
            is_label = "future_translation_m" in agent_forecast
            if not is_label and agent_forecast["detection_score"] < only_show_detection_scores_above:
                continue
            if skip_static_predictions and not is_label and label_forecasts:
                closest_label = label_forecasts[np.argmin([np.linalg.norm(xy - a["current_translation_m"]) for a in label_forecasts])]
                # check if prediction is static
                agent_forecast["velocity_m_per_s"] = np.zeros_like(agent_forecast["prediction_m"][0])
                prediction_type = trajectory_type(agent_forecast, CATEGORY_TO_VELOCITY_M_PER_S)[np.argmax(agent_forecast["score"])]
                label_type = trajectory_type(closest_label, CATEGORY_TO_VELOCITY_M_PER_S)
                if label_type == "static" and prediction_type == "static":
                    continue

            if is_label:
                plt.scatter(*xy, c=color_lookup("LABEL"), s=1)
                plt.gca().add_patch(get_rectangle(*xy, *agent_forecast["size"][:2], agent_forecast["yaw"], fill=False, color=color_lookup("LABEL"), alpha=0.3))
                path = np.concatenate([xy.reshape(1, 2), agent_forecast["future_translation_m"]])
                plt.plot(path[:, 0], path[:, 1], c=color_lookup("LABEL"))
            else:
                plt.gca().add_patch(get_rectangle(*xy, *agent_forecast["size"][:2], agent_forecast["yaw"], fill=False, color=color_lookup(agent_forecast["name"])))
                plt.scatter(*xy, c=color_lookup(agent_forecast["name"]), s=1)
                if only_show_most_likely:
                    futures = [(agent_forecast["prediction_m"][np.argmax(agent_forecast["score"])], agent_forecast["score"].max())]
                else:
                    futures = zip(agent_forecast["prediction_m"], agent_forecast["score"])

                for future, probability in futures:
                    path = np.concatenate([xy.reshape(1, 2), future])
                    plt.plot(path[:, 0], path[:, 1], c=color_lookup(agent_forecast["name"]), alpha=probability)

        # plot legend
        plt.scatter([-1e5], [-1e5], c=color_lookup("LABEL"), label="label future")
        plt.scatter([-1e5], [-1e5], c=color_lookup("REGULAR_VEHICLE"), label="vehicles")
        plt.scatter([-1e5], [-1e5], c=color_lookup("PEDESTRIAN"), label="pedestrian")
        plt.scatter([-1e5], [-1e5], c=color_lookup("BICYCLIST"), label="wheeled devices")
        plt.scatter([-1e5], [-1e5], c=color_lookup("OTHER"), label="other")
        plt.legend(loc="lower left")
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
        image = Image.open(img_buf)
        images.append(image)
        plt.clf()
        plt.close()


    animation = ArtistAnimation(plt.figure(figsize=(10, 10)), [[plt.imshow(im, animated=True)] for im in images], interval=500, blit=True)
    return animation

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--forecast_predictions", required=True)
    argparser.add_argument("--seq_id", required=True, type=str)
    argparser.add_argument("--split", default="val", choices=["val", "test"])
    argparser.add_argument("--out", default="scene.png", type=str)
    config = argparser.parse_args()

    # seq_id = "95bf6003-7068-3a78-a0c0-9e470a06e60f"
    # forecasts = load("/home/ashen3/LT3D/results/av2-val/lstm_forecaster/forecast_predictions.pkl")
    # labels = convert_forecast_labels(load("/home/ashen3/LT3D/dataset/av2-val/labels.pkl"))
    label_path = os.path.join("dataset", f"av2-{config.split}", "labels.pkl")
    animation = make_animation(
        config.seq_id,
        load(config.forecast_predictions),
        convert_forecast_labels(load(label_path)),
        PATHS["av2"][config.split]["dataset_dir"],
    )
    animation.save(config.out)
