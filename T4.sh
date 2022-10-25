#bash tools/dist_train.sh configs/centerpoint/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2.py 8 --no-validate
bash tools/dist_test.sh configs/centerpoint/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2.py work_dirs/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/latest.pth 8 --cached work_dirs/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/predictions.pkl --eval mAP --metric_type standard --predictions work_dirs/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/val_detections.csv --ground_truth work_dirs/centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_150m_wide_tta_20e_av2/val_gt.csv 


