PATHS = {
    "av2": {
        "train": {
            "infos": "/ssd0/nperi/Sensor/av2_mmdet3d_trainval/av2_infos_train.pkl",
            "dataset_dir": "/ssd0/nperi/Sensor/train",
        },
        "val": {
            "prediction": "/home/nperi/Workspace/Deployment/LT3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_av2/val_detections.csv",
            #"prediction": "/home/nperi/Workspace/Deployment/LT3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_av2/predictions.pkl",
            "infos": "/ssd0/nperi/Sensor/av2_mmdet3d_trainval/av2_infos_val.pkl",
            "dataset_dir": "/ssd0/nperi/Sensor/val",
        },
        "test": {
            "prediction": "../work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/test_predictions.pkl",
            "infos": "/ssd0/nperi/Sensor/av2_mmdet3d_trainval/av2_infos_test.pkl",
            "dataset_dir": "/ssd0/nperi/Sensor/test",
        },
    },
    "nuscenes": {
        "val": {
            "prediction": "work_dirs/hv_pointpillars_fpn_sbn-all_4x8_2x_hierarchy_nus/predictions.pkl",
            "infos": "/ssd0/nperi/nuScenes/nusc_mmdet3d_trainval/nuscenes_infos_val.pkl",
            "dataset_dir": "/ssd0/nperi/nuScenes",
        },
    },
}
