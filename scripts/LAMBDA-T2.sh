#python tools/create_data.py av2 --root-path /ssd0/nperi/Sensor/ --out-dir /ssd0/nperi/Sensor/av2_mmdet3d_trainval/ --max-sweeps 5 --extra-tag av2
bash tools/dist_train.sh configs/pointpillars/mrp/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_50m_av2.py 8 --no-validate
bash tools/dist_test.sh configs/pointpillars/mrp/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_50m_av2.py work_dirs/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_50m_av2/latest.pth 8 --out work_dirs/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_50m_av2/predictions.pkl --eval mAP --metric_type standard