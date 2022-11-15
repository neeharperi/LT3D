import os 
import pandas as pd 
import numpy as np
from pathlib import Path
from tqdm import tqdm 


model = "pointpillars"
near_far = True 
ranges = [(0, 50), (50, 100), (100, 150)]
sample_rate = [2, 2, 2]

print(model, near_far)

if model == "pointpillars":
    range_type = "ensemble_50-0125_100-025_150-05{}".format("_nf" if near_far else "")
    out_path = "work_dirs/pointpillars_{}_av2/".format(range_type)
    ground_truth = "work_dirs/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_av2/val_gt.csv"

    detections = [
    "work_dirs/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_av2/val_detections.csv",
    "work_dirs/hv_pointpillars_025voxel_fpn_sbn-all_4x8_2x_100m_av2/val_detections.csv",
    "work_dirs/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_150m_av2/val_detections.csv",
    ]

elif model == "centerpoint":
    range_type = "ensemble_50-0075_100-01_150-02{}".format("_nf" if near_far else "")
    out_path = "work_dirs/centerpoint_{}_av2/".format(range_type)
    ground_truth = "work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_tta_20e_av2/val_gt.csv"

    detections = [
    "work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_tta_20e_av2/val_detections.csv",
    "work_dirs/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_100m_wide_tta_20e_av2/val_detections.csv",
    "work_dirs/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/val_detections.csv",
    ]

else:
    assert False, "Invalid Model"

os.makedirs(out_path, exist_ok=True)
groundTruthDataFrame = pd.read_csv(ground_truth, usecols=['tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 'height_m',
                                                            'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'score', 'num_interior_pts',
                                                            'log_id', 'track_uuid', 'timestamp_ns', 'category'])

data = groundTruthDataFrame[["log_id", 'timestamp_ns']]      
sequence = {}
for row in data.iterrows():
    log_id = row[1]["log_id"]
    timestamp = row[1]["timestamp_ns"]

    if log_id not in sequence:
        sequence[log_id] = []

    if timestamp not in sequence[log_id]:
        sequence[log_id].append(timestamp)


for log_id in sequence:
    sequence[log_id] = sorted(sequence[log_id])

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

for dets, dist, rate in zip(detections, ranges, sample_rate):
    preds = pd.read_csv(dets) 
    min_dist, max_dist = dist
    
    pred_sequence = {}
    for pred in tqdm(preds.iterrows(), total=preds.shape[0]):   
        l2 = np.linalg.norm([pred[1]['tx_m'], pred[1]['ty_m']])
        if l2 > min_dist and l2 <= max_dist:
            log_id = pred[1]['log_id']
            timestamp = pred[1]['timestamp_ns']

            ps = {'tx_m' : pred[1]['tx_m'],
                'ty_m' : pred[1]['ty_m'],
                'tz_m' : pred[1]['tz_m'],
                'qw' : pred[1]['qw'],
                'qx' : pred[1]['qx'],
                'qy' : pred[1]['qy'],
                'qz' : pred[1]['qz'],
                'length_m' : pred[1]['length_m'],
                'width_m' : pred[1]['width_m'],
                'height_m' : pred[1]['height_m'],
                'vx' : pred[1]['vx'],
                'vy' : pred[1]['vy'],
                'category' : pred[1]['category'],
                'score' : pred[1]['score'],
                'log_id' : pred[1]['log_id'],
                'timestamp_ns' : pred[1]['timestamp_ns']
            }

            if log_id not in pred_sequence:
                pred_sequence[log_id] = {}

            if timestamp not in pred_sequence[log_id]:
                pred_sequence[log_id][timestamp] = []

            pred_sequence[log_id][timestamp].append(ps)

    for log_id in sequence:
        for i, timestamp in enumerate(sequence[log_id]):
            if log_id not in pred_sequence:
                continue 

            if timestamp not in pred_sequence[log_id]:
                continue 
            
            if i % rate == 0:
                last_timestamp = timestamp

                for pred in pred_sequence[log_id][timestamp]:
                    predictions['tx_m'].append(pred['tx_m'])
                    predictions['ty_m'].append(pred['ty_m'])
                    predictions['tz_m'].append(pred['tz_m'])

                    predictions['qw'].append(pred['qw'])
                    predictions['qx'].append(pred['qx'])
                    predictions['qy'].append(pred['qy'])
                    predictions['qz'].append(pred['qz'])

                    predictions['length_m'].append(pred['length_m'])
                    predictions['width_m'].append(pred['width_m'])
                    predictions['height_m'].append(pred['height_m'])

                    predictions['vx'].append(pred['vx'])
                    predictions['vy'].append(pred['vy'])

                    predictions['category'].append(pred['category'])
                    predictions['score'].append(pred['score'])

                    predictions['log_id'].append(pred['log_id'])
                    predictions['timestamp_ns'].append(pred['timestamp_ns'])
            else:
                if near_far is True:
                    time_delta = 1e-9*(timestamp - last_timestamp)
                else:
                    time_delta = 0
                    last_timestamp = timestamp

                for pred in pred_sequence[log_id][last_timestamp]:
                    l2 = np.linalg.norm([pred['tx_m'] + time_delta * pred['vx'], pred['ty_m'] + time_delta * pred['vy']])
                    if l2 > min_dist and l2 <= max_dist:
                        predictions['tx_m'].append(pred['tx_m'] + time_delta * pred['vx'])
                        predictions['ty_m'].append(pred['ty_m'] + time_delta * pred['vy'])
                        predictions['tz_m'].append(pred['tz_m'])

                        predictions['qw'].append(pred['qw'])
                        predictions['qx'].append(pred['qx'])
                        predictions['qy'].append(pred['qy'])
                        predictions['qz'].append(pred['qz'])

                        predictions['length_m'].append(pred['length_m'])
                        predictions['width_m'].append(pred['width_m'])
                        predictions['height_m'].append(pred['height_m'])

                        predictions['vx'].append(pred['vx'])
                        predictions['vy'].append(pred['vy'])

                        predictions['category'].append(pred['category'])
                        predictions['score'].append(pred['score'])

                        predictions['log_id'].append(pred['log_id'])
                        predictions['timestamp_ns'].append(pred['timestamp_ns'])

predictionsDataFrame = pd.DataFrame.from_dict(predictions)

pd.DataFrame.to_csv(predictionsDataFrame, out_path + "/val_detections.csv", index=False)
pd.DataFrame.to_csv(groundTruthDataFrame, out_path + "/val_gt.csv", index=False)
