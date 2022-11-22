#python tools/create_data.py av2 --root-path /ssd0/nperi/Sensor/ --out-dir /ssd0/nperi/Sensor/av2_mmdet3d_trainval/ --max-sweeps 5 --extra-tag av2
bash tools/dist_train.sh configs/pointpillars/mrp_lite/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_lite_av2.py 8 --no-validate
bash tools/dist_train.sh configs/pointpillars/mrp_lite/hv_pointpillars_025voxel_fpn_sbn-all_4x8_2x_100m_lite_av2.py 8 --no-validate
bash tools/dist_train.sh configs/pointpillars/mrp_lite/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_150m_lite_av2.py 8 --no-validate

bash tools/dist_train.sh configs/pointpillars/mrp_interval/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_interval_av2.py 8 --no-validate
bash tools/dist_train.sh configs/pointpillars/mrp_interval/hv_pointpillars_025voxel_fpn_sbn-all_4x8_2x_100m_interval_av2.py 8 --no-validate
bash tools/dist_train.sh configs/pointpillars/mrp_interval/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_150m_interval_av2.py 8 --no-validate
