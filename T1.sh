python tools/create_data.py av2 --root-path /ssd0/nperi/Sensor/ --out-dir /ssd0/nperi/Sensor/av2_mmdet3d_trainval/ --extra-tag av2
bash tools/dist_train.sh configs/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2.py 8 --no-validate
bash tools/dist_test.sh configs/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2.py work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/latest.pth 8 --out work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/predictions.pkl --eval mAP --metric_type standard


python tools/train.py configs/centerpoint/centerpoint_0075pillar_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_nus.py --no-validate
