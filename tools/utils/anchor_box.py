from av2.datasets.sensor.splits import TRAIN
from av2.utils.io import read_feather
import mmcv
import os
import numpy as np

from tqdm import tqdm 

root_path = '/ssd0/nperi/Sensor/'
classes = ['REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
            'PEDESTRIAN', 'WHEELED_RIDER', 'BICYCLE', 'BICYCLIST', 'MOTORCYCLE', 'MOTORCYCLIST', 'WHEELED_DEVICE', 'WHEELED_RIDER', 'WHEELCHAIR', 'STROLLER', 'DOG',
            'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MESSAGE_BOARD_TRAILER']

gt_lwh = []
gt_z = []
gt_names = [] 

for log_id in tqdm(TRAIN):
    split = "train"

    log_dir = "{root_path}/{split}/{log_id}".format(root_path=root_path, split=split, log_id=log_id)
    lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
    annotations_path = "{log_dir}/annotations.feather".format(log_dir=log_dir)
    annotations = read_feather(annotations_path)

    mmcv.check_file_exist(annotations_path)

    for i, filename in enumerate(os.listdir(lidar_paths)):
        timestamp_ns = int(filename.split(".")[0])
        lidar_path = "{log_dir}/sensors/lidar/{timestamp_ns}.feather".format(log_dir=log_dir,timestamp_ns=timestamp_ns)

        mmcv.check_file_exist(lidar_path)
        
        curr_annotations = annotations[annotations["timestamp_ns"] == timestamp_ns]
        curr_annotations = curr_annotations[curr_annotations["num_interior_pts"] > 0]


        for annotation in curr_annotations.iterrows():
            z = np.array([annotation[1]["tz_m"]])
            lwh = np.array([annotation[1]["length_m"], annotation[1]["width_m"], annotation[1]["height_m"]])
            class_name = annotation[1]["category"]

            gt_z.append(z)
            gt_lwh.append(lwh) 
            gt_names.append(class_name)
            
gt_names = np.array(gt_names)
gt_z = np.array(gt_z)
gt_lwh = np.array(gt_lwh)

for cname in classes:
    keep = np.array(gt_names == cname) 
    z_class = gt_z[keep]
    
    print("[point_cloud_range[0], point_cloud_range[1], {z}, point_cloud_range[3], point_cloud_range[4], {z}] #{cname}".format(z=np.mean(z_class), cname=cname))
        
for cname in classes:
    keep = np.array(gt_names == cname) 
    lwh_class = gt_lwh[keep]
    
    print("[{l}, {w}, {h}] #{cname}".format(l=np.mean(lwh_class, axis=0)[0], w=np.mean(lwh_class, axis=0)[1], h=np.mean(lwh_class, axis=0)[2], cname=cname))

    
