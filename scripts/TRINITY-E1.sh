python tools/test.py configs/pointpillars/mrp_interval/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_150m_interval_av2.py work_dirs/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_150m_interval_av2/epoch_24.pth --out work_dirs/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_150m_interval_av2/predictions.pkl --eval mAP --metric_type standard
python tools/test.py configs/pointpillars/mrp_interval/hv_pointpillars_025voxel_fpn_sbn-all_4x8_2x_100m_interval_av2.py work_dirs/hv_pointpillars_025voxel_fpn_sbn-all_4x8_2x_100m_interval_av2/epoch_24.pth --out work_dirs/hv_pointpillars_025voxel_fpn_sbn-all_4x8_2x_100m_interval_av2/predictions.pkl --eval mAP --metric_type standard
python tools/test.py configs/pointpillars/mrp_interval/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_interval_av2.py work_dirs/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_interval_av2/epoch_24.pth --out work_dirs/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_interval_av2/predictions.pkl --eval mAP --metric_type standard

python tools/test.py configs/pointpillars/mrp_lite/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_150m_lite_av2.py work_dirs/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_150m_lite_av2/epoch_24.pth --out work_dirs/hv_pointpillars_05voxel_fpn_sbn-all_4x8_2x_150m_lite_av2/predictions.pkl --eval mAP --metric_type standard
python tools/test.py configs/pointpillars/mrp_lite/hv_pointpillars_025voxel_fpn_sbn-all_4x8_2x_100m_lite_av2.py work_dirs/hv_pointpillars_025voxel_fpn_sbn-all_4x8_2x_100m_lite_av2/epoch_24.pth --out work_dirs/hv_pointpillars_025voxel_fpn_sbn-all_4x8_2x_100m_lite_av2/predictions.pkl --eval mAP --metric_type standard
python tools/test.py configs/pointpillars/mrp_lite/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_lite_av2.py work_dirs/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_lite_av2/epoch_24.pth --out work_dirs/hv_pointpillars_0125voxel_fpn_sbn-all_4x8_2x_50m_lite_av2/predictions.pkl --eval mAP --metric_type standard