import os 
import pandas as pd 
import numpy as np
from pathlib import Path
from tqdm import tqdm 
from 
range_type = "ensemble_50-0075_100-01_150-02"
out_path = "work_dirs/centerpoint_{}_av2/".format(range_type)
ground_truth = "work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_hierarchy_tta_20e_av2/val_gt.csv"

detections = ["work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_hierarchy_tta_20e_av2/val_detections.csv",
              "work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_hierarchy_tta_20e_av2/val_detections.csv",
              "work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_hierarchy_tta_20e_av2/val_detections.csv"]
ranges = [(0, 50), (50, 100), (100, 150)]

groundTruthDataFrame = pd.read_csv(ground_truth, usecols=['tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 'height_m',
                                                            'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'score', 'num_interior_pts',
                                                            'log_id', 'track_uuid', 'timestamp_ns', 'category'])
                                                            
#########################################
predictions = {"tx_m" : [],
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
                "timestamp_ns" : [],
                "category" : []}

for dets, dist in zip(detections, ranges):
    preds = pd.read_csv(dets) 
    min_dist, max_dist = dist

    for pred in tqdm(preds.iterrows(), total=preds.shape[0]):   
        l2 = np.linalg.norm([pred[1]['tx_m'], pred[1]['ty_m']])
        if l2 > min_dist and l2 <= max_dist:
            predictions['tx_m'].append(pred[1]['tx_m'])
            predictions['ty_m'].append(pred[1]['ty_m'])
            predictions['tz_m'].append(pred[1]['tz_m'])

            predictions['qw'].append(pred[1]['qw'])
            predictions['qx'].append(pred[1]['qx'])
            predictions['qy'].append(pred[1]['qy'])
            predictions['qz'].append(pred[1]['qz'])

            predictions['length_m'].append(pred[1]['length_m'])
            predictions['width_m'].append(pred[1]['width_m'])
            predictions['height_m'].append(pred[1]['height_m'])

            predictions['vx'].append(pred[1]['vx'])
            predictions['vy'].append(pred[1]['vy'])

            predictions['category'].append(pred[1]['category'])
            predictions['score'].append(pred[1]['score'])

            predictions['log_id'].append(pred[1]['log_id'])
            predictions['timestamp_ns'].append(pred[1]['timestamp_ns'])

predictionsDataFrame = pd.DataFrame.from_dict(predictions)
pd.DataFrame.to_csv(predictionsDataFrame, out_path + "/val_detections.csv", index=False)
pd.DataFrame.to_csv(groundTruthDataFrame, out_path + "/val_gts.csv", index=False)