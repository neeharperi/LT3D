bash tools/dist_train.sh configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_tta_20e_av2.py 8 --no-validate
bash tools/dist_test.sh configs/centerpoint/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_tta_20e_av2.py work_dirs/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_tta_20e_av2/latest.pth 8 --out work_dirs/centerpoint_02voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_tta_20e_av2/predictions.pkl --eval mAP --metric_type standard

