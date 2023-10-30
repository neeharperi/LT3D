import os
from argparse import ArgumentParser
from data.paths import PATHS
from data.utils import *
from forecasting.linear_forecaster import forecast
from forecasting.lstm.lstm import LSTMModel, generate_forecasts_from_model
from av2.evaluation.forecasting.constants import CATEGORY_TO_VELOCITY_M_PER_S
from pprint import pprint 

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="av2", choices=["av2", "nuscenes"])
    argparser.add_argument("--split", default="val", choices=["val", "test"])
    argparser.add_argument("--track_predictions", required=True)
    argparser.add_argument("--forecaster", default="linear_forecaster", choices=["linear_forecaster", "lstm_forecaster"])
    argparser.add_argument("--device", default="cuda")
    
    argparser.add_argument("--time_delta", type=float, default=0.5)
    argparser.add_argument("--num_timesteps", type=int, default=6)
    argparser.add_argument("--top_k", type=int, default=5)
    argparser.add_argument("--ego_distance_threshold_m", type=int, default=50)
    config = argparser.parse_args()

    paths = PATHS[config.dataset][config.split]
    if config.dataset == "av2":
        from data.av2_adapter import av2_load_predictions_and_labels
        class_names = AV2_CLASS_NAMES
        class_velocity = CATEGORY_TO_VELOCITY_M_PER_S
        
        predictions, labels = av2_load_predictions_and_labels(
            paths["prediction"], paths["infos"], paths["dataset_dir"]
        )
        
    elif config.dataset == "nuscenes":
        from nuscenes import NuScenes
        from data.nuscenes_adapter import nuscenes_load_predictions_and_labels

        nusc_version = "v1.0-trainval"
        nusc = NuScenes(nusc_version, dataroot=paths["dataset_dir"])
        class_names = NUSCENES_CLASS_NAMES
        class_velocity = nuscenes_velocity

        predictions, labels = nuscenes_load_predictions_and_labels(
            paths["prediction"], paths["infos"], nusc
        )
            
    _dataset_dir = os.path.join("results", f"{config.dataset}-{config.split}")
    results_dir = os.path.join(_dataset_dir, f"{config.forecaster}")
    track_predictions = load(config.track_predictions)

    if config.forecaster == "linear_forecaster":
        VELOCITY_SCALAR = [1.0, 1.2, 1.4, 0.8, 0.6]
        forecasts = forecast(
            track_predictions,
            velocity_scalar=VELOCITY_SCALAR,
            num_timesteps=config.num_timesteps,
            time_delta=config.time_delta,
        )
    elif config.forecaster == "lstm_forecaster":
        import torch

        model = torch.load(os.path.join("models", config.dataset, "lstm.pt"))
        # run inference
        forecasts = generate_forecasts_from_model(
            model,
            track_predictions,
            config.num_timesteps,
            config.top_k,
            config.device,
        )
    else:
        raise Exception(f"Forecaster {config.forecaster} not supported")
    
    outputs_dir = os.path.join(
        "results", f"{config.dataset}-{config.split}", config.forecaster, "outputs"
    )
    print(f"Saving forecast predictions to {outputs_dir}")
    save(labels, os.path.join("dataset", f"{config.dataset}-{config.split}", "labels.pkl"))
    save(forecasts, os.path.join(results_dir, "forecast_predictions.pkl"))

    if config.split != "test":
        if config.dataset == "av2":
            from av2.evaluation.forecasting.eval import evaluate
            res =  evaluate(forecasts, labels, config.top_k, config.ego_distance_threshold_m, paths["dataset_dir"])
            mAP_F = np.nanmean([metrics["mAP_F"] for traj_metrics in res.values() for metrics in traj_metrics.values()])
            ADE = np.nanmean([metrics["ADE"] for traj_metrics in res.values() for metrics in traj_metrics.values()])
            FDE = np.nanmean([metrics["FDE"] for traj_metrics in res.values() for metrics in traj_metrics.values()])
            print("mAP_F: {}, ADE: {}, FDE: {}".format(mAP_F, ADE, FDE))
        elif config.dataset == "nuscenes":
            raise Exception(f"Not Implemented Yet")
                
        save(res, os.path.join(outputs_dir, "res.pkl"))
