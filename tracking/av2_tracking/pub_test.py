from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from pub_tracker import PubTracker as Tracker
import pandas as pd 
from tqdm import tqdm 
import time
from pathlib import Path
from av2.datasets.sensor.sensor_dataloader import read_city_SE3_ego

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir")
    parser.add_argument("--root_dir", default="/home/ubuntu/Workspace/Data/Sensor/val")
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--max_age", type=int, default=3)

    args = parser.parse_args()

    return args

def convert_tracks(dataFrame, reference_SE3_ego_t, thresh=0.1):
    res = [] 

    for rowTable in dataFrame.iterrows():
        score = rowTable[1]["score"]

        if score < thresh: 
            continue 
        
        translation = [rowTable[1]["tx_m"], rowTable[1]["ty_m"], rowTable[1]["tz_m"]]
        translation = reference_SE3_ego_t.transform_from(translation)

        anno = {
            "sample_token": rowTable[1]["log_id"],
            "timestamp_ns": rowTable[1]["timestamp_ns"],
            "translation": translation.tolist(),
            "size": [rowTable[1]["length_m"], rowTable[1]["width_m"], rowTable[1]["height_m"]],
            "rotation": [rowTable[1]["qw"], rowTable[1]["qx"], rowTable[1]["qy"], rowTable[1]["qz"]],
            "velocity": [rowTable[1]["vx"], rowTable[1]["vy"]],
            "detection_name": rowTable[1]["category"],
            "detection_score": rowTable[1]["score"],
        }
        res.append(anno)

    return res

def main():
    args = parse_args()

    tracker = Tracker(max_age=args.max_age, hungarian=args.hungarian)
    predictions = pd.read_csv(args.work_dir + "/val_detections.csv")

    av2_anno = {"tx_m" : [],
                "ty_m" : [],
                "tz_m" : [],
                "length_m" : [],
                "width_m" : [],
                "height_m" : [],
                "qw" : [],
                "qx" : [],
                "qy" : [],
                "qz" : [],
                "vx" : [],
                "vy" : [],
                "score" : [],
                "log_id" : [],
                "track_uuid" : [],
                "timestamp_ns" : [],
                "category" : []}

    logs = set(predictions["log_id"].tolist())
    for log_id in tqdm(logs):
        log_preds = predictions[predictions["log_id"] == log_id]
        timestamps = sorted(set(log_preds["timestamp_ns"].tolist()))

        log_dir = Path(args.root_dir + "/{}".format(log_id))
        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir)
        city_SE3_ego_reference = timestamp_city_SE3_ego_dict[timestamps[0]]

        for i, timestamp in enumerate(timestamps):
            if i == 0:
                tracker.reset()
                last_time_stamp = timestamp

            city_SE3_ego_t = timestamp_city_SE3_ego_dict[timestamp]
            reference_SE3_ego_t = city_SE3_ego_reference.inverse().compose(city_SE3_ego_t)

            time_lag = (timestamp - last_time_stamp) 
            last_time_stamp = timestamp

            timestamp_preds = log_preds[log_preds["timestamp_ns"] == timestamp]
            track_preds = convert_tracks(timestamp_preds, reference_SE3_ego_t)

            outputs = tracker.step_centertrack(track_preds, time_lag)

   
            reference_SE3_t_ego = city_SE3_ego_t.inverse().compose(city_SE3_ego_reference)

            for item in outputs:
                if item['active'] == 0:
                    continue 
                
                translation = [item['translation'][0], item['translation'][1], item['translation'][2]]
                translation = reference_SE3_t_ego.transform_from(translation)

                av2_anno['tx_m'].append(translation[0])
                av2_anno['ty_m'].append(translation[1])
                av2_anno['tz_m'].append(translation[2])

                av2_anno['qw'].append(item["rotation"][0])
                av2_anno['qx'].append(item["rotation"][1])
                av2_anno['qy'].append(item["rotation"][2])
                av2_anno['qz'].append(item["rotation"][3])

                av2_anno['length_m'].append(item["size"][0])
                av2_anno['width_m'].append(item["size"][1])
                av2_anno['height_m'].append(item["size"][2])

                av2_anno['vx'].append(item["velocity"][0])
                av2_anno['vy'].append(item["velocity"][1])

                av2_anno['category'].append(item["detection_name"])
                av2_anno['score'].append(item["detection_score"])
                av2_anno['track_uuid'].append(item["tracking_id"])

                av2_anno['log_id'].append(item["sample_token"])
                av2_anno['timestamp_ns'].append(item["timestamp_ns"])

    trackDataFrame = pd.DataFrame.from_dict(av2_anno)
    trackDataFrame.to_csv(args.work_dir + "/val_tracks.csv")
    
if __name__ == '__main__':
    main()
