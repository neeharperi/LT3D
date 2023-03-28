import argparse 
import pickle
import utils
from tqdm import tqdm
import numpy as np 

velocity_scalar = [1.0, 1.2, 1.4, 0.8, 0.6]

def forecast(tracks):
    res = {}
    for seq_id in tqdm(tracks.keys()):
        res[seq_id] = {}
        
        for agents in tracks[seq_id]:
            timestamp = agents["timestamp_ns"]
            res[seq_id][timestamp] = []
        
            for i in range(agents["track_id"].shape[0]):
                res[seq_id][timestamp].append({"timestep_ns" : timestamp, 
                                                "current_translation" : agents["translation"][i][:2],
                                                "detection_score" : agents["score"][i], 
                                                "size" : agents["size"][i], 
                                                "label" : agents["label"][i], 
                                                "name" : agents["name"][i], 
                                                "prediction" : constant_velocity(agents["translation"][i], agents["velocity"][i]),
                                                "score" : 5 * [agents["score"][i]], 
                                                "instance_id" : agents["track_id"][i],
                                            })
                
    return res

def constant_velocity(center, velocity):
    num_timesteps = utils.num_timesteps
    time_delta = utils.time_delta
    
    forecasts = []
    for i in range(5):
        forecast = []
        
        for j in range(num_timesteps):
            forecast.append(center[:2] + velocity_scalar[i] * j * time_delta * velocity[:2])
            
        forecasts.append(np.array(forecast))
        
    return np.array(forecasts)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", default="nuscenes", choices=["av2", "nuscenes"])
    argparser.add_argument("--tracks", default="sample/track_predictions.pkl")
    argparser.add_argument("--out", default="sample/constant_velocity.pkl")

    args = argparser.parse_args()
    tracks = pickle.load(open(args.tracks, "rb"))

    if args.dataset == "av2":
        class_names = utils.av2_classes
        class_velocity = utils.av2_velocity

    if args.dataset == "nuscenes":
        class_names = utils.nus_classes
        class_velocity = utils.nus_velocity 

    res = forecast(tracks)
    pickle.dump(res, open(args.out, "wb"))