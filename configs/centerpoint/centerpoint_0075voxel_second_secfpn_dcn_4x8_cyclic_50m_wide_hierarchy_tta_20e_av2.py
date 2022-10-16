dataset_type = 'AV2Dataset'
data_root = '/ssd0/nperi/Sensor/'
VERSION = "av2_mmdet3d_trainval"
file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

WIDE = True
WIDE_DIM=512

SAMPLER_TYPE = "STANDARD"

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -3, 54, 54, 3]
sparse_shape = [(point_cloud_range[5] / voxel_size[2]) + 1, point_cloud_range[4] / voxel_size[1], point_cloud_range[3] / voxel_size[0]]
grid_size = [point_cloud_range[3] / voxel_size[0], point_cloud_range[4] / voxel_size[1], point_cloud_range[5] / voxel_size[2]]

file_client_args = dict(backend='disk')
class_names = [
    'REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG'
]

total_class_names = ['REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER', 'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 
                     'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER', 
                     'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG', "VEHICLE", "VULNERABLE", "MOVABLE", "OBJECT"]

TASK_NAMES = {"STANDARD": ['REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER', 'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 
                            'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER', 
                            'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG'],
              "GROUP": ["VEHICLE", "VULNERABLE", "MOVABLE"],
              "ALL" : ["OBJECT"]
             }

CLASS_MAPPING = {"STANDARD": [['REGULAR_VEHICLE'], ['PEDESTRIAN'], ['BICYCLIST'], ['MOTORCYCLIST'], ['WHEELED_RIDER'], ['BOLLARD'], ['CONSTRUCTION_CONE'], ['SIGN'], ['CONSTRUCTION_BARREL'], ['STOP_SIGN'], 
                                ['MOBILE_PEDESTRIAN_CROSSING_SIGN'], ['LARGE_VEHICLE'], ['BUS'], ['BOX_TRUCK'], ['TRUCK'], ['VEHICULAR_TRAILER'], ['TRUCK_CAB'], ['SCHOOL_BUS'], ['ARTICULATED_BUS'], ['MESSAGE_BOARD_TRAILER'], 
                                ['BICYCLE'], ['MOTORCYCLE'], ['WHEELED_DEVICE'], ['WHEELCHAIR'], ['STROLLER'], ['DOG']],
                 "GROUP" : [['REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS'], 
                            ['PEDESTRIAN', 'WHEELED_RIDER', 'BICYCLE', 'MOTORCYCLE', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG'], 
                            ['BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MESSAGE_BOARD_TRAILER']],
                 "ALL" : [['REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER', 'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 
                                'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER', 
                                'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG']]
                 }

NMS = ["W", "W", "W", "W", "W", "W", "W", "W", "W", "W", 
       "W", "W", "W", "W", "W", "W", "W", "W", "W", "W", 
       "W", "W", "W", "W", "W", "W", "X", "X", "X", "X"]
#NMS = ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", 
#     "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", 
#      "A", "A", "A", "A", "A", "A", "X", "X", "X", "X"]

GROUP = ['REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER', 'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 
        'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER', 
        'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG', "VEHICLE", "VULNERABLE", "MOVABLE", "OBJECT"]
#GROUP = ['VEHICLE', 'VULNERABLE', 'VULNERABLE', 'VULNERABLE', 'VULNERABLE', 'MOVABLE', 'MOVABLE', 'MOVABLE', 'MOVABLE', 'MOVABLE', 
#        'MOVABLE', 'VEHICLE', 'VEHICLE', 'VEHICLE', 'VEHICLE', 'VEHICLE', 'VEHICLE', 'VEHICLE', 'VEHICLE', 'MOVABLE', 
#        'VULNERABLE', 'VULNERABLE', 'VULNERABLE', 'VULNERABLE', 'VULNERABLE', 'VULNERABLE', "VEHICLE", "VULNERABLE", "MOVABLE", "OBJECT"]
GROUP = [total_class_names.index(g) for g in GROUP]

model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(max_num_points=10, voxel_size=voxel_size, max_voxels=(90000, 120000), point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=6),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=6,
        sparse_shape=[31, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=128,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[dict(num_class=30, class_names=['REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER', 'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 
                                                'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER', 
                                                'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG', 'VEHICLE', 'VULNERABLE', 'MOVABLE', 'OBJECT']),
              ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=WIDE_DIM,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-54, -54, -3, 54, 54, 3],
            max_num=16384,
            score_threshold=0.01,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9),
        separate_head=dict(
            type='DCNSeparateHead',
            dcn_config=dict(
                type='DCN',
                in_channels=WIDE_DIM,
                out_channels=WIDE_DIM,
                kernel_size=3,
                padding=1,
                groups=4),
            head_conv=WIDE_DIM,
            init_bias=-2.19,
            final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 30],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            post_center_range=[-54, -54, -3, 54, 54, 3],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.01,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=500,
            nms_thr=0.2,
            wide=WIDE,
            nms=NMS, 
            group=GROUP)))

db_sampler = dict(
data_root=data_root,
info_path=data_root + '{}/av2_dbinfos_train.pkl'.format(VERSION),
rate=1.0,
sampler_type=SAMPLER_TYPE, 
task_names=TASK_NAMES,
class_mapping=CLASS_MAPPING,
prepare=dict(
    filter_by_difficulty=[-1],
    filter_by_min_points=dict(
        REGULAR_VEHICLE=5,
        PEDESTRIAN=5,
        BICYCLIST=5,
        MOTORCYCLIST=5,
        WHEELED_RIDER=5,
        BOLLARD=5,
        CONSTRUCTION_CONE=5,
        SIGN=5,
        CONSTRUCTION_BARREL=5,
        STOP_SIGN=5,
        MOBILE_PEDESTRIAN_CROSSING_SIGN=5,
        LARGE_VEHICLE=5,
        BUS=5,
        BOX_TRUCK=5,
        TRUCK=5,
        VEHICULAR_TRAILER=5,
        TRUCK_CAB=5,
        SCHOOL_BUS=5,
        ARTICULATED_BUS=5,
        MESSAGE_BOARD_TRAILER=5,
        BICYCLE=5,
        MOTORCYCLE=5,
        WHEELED_DEVICE=5,
        WHEELCHAIR=5,
        STROLLER=5,
        DOG=5,
        )),
classes=class_names,
sample_groups=dict(
    REGULAR_VEHICLE=2,
    PEDESTRIAN=2,
    BICYCLIST=5,
    MOTORCYCLIST=5,
    WHEELED_RIDER=6,
    BOLLARD=3,
    CONSTRUCTION_CONE=3,
    SIGN=4,
    CONSTRUCTION_BARREL=3,
    STOP_SIGN=3,
    MOBILE_PEDESTRIAN_CROSSING_SIGN=7,
    LARGE_VEHICLE=4,
    BUS=4,
    BOX_TRUCK=4,
    TRUCK=5,
    VEHICULAR_TRAILER=5,
    TRUCK_CAB=5,
    SCHOOL_BUS=6,
    ARTICULATED_BUS=6,
    MESSAGE_BOARD_TRAILER=7,
    BICYCLE=3,
    MOTORCYCLE=5,
    WHEELED_DEVICE=4,
    WHEELCHAIR=7,
    STROLLER=6,
    DOG=5,
    ),
points_loader=dict(
    type='LoadPointsFromFileFeather',
    coord_type='LIDAR',
    load_dim=6,
    use_dim=[0, 1, 2, 3, 4, 5],
    shift_height=False,
    use_color=False,
    file_client_args=file_client_args))

train_pipeline = [
dict(
    type='LoadPointsFromFileFeather',
    coord_type='LIDAR',
    load_dim=6,
    use_dim=[0, 1, 2, 3, 4, 5],
    shift_height=False,
    use_color=False,
    file_client_args=file_client_args),
dict(
    type='LoadPointsFromMultiSweepsFeather',
    coord_type="LIDAR",
    sweeps_num=5,
    load_dim=6,
    use_dim=[0, 1, 2, 3, 4, 5],
    pad_empty_sweeps=True,
    remove_close=True,
    test_mode=False,
    shift_height=False,
    use_color=False,
    file_client_args=file_client_args),
dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
dict(type='ObjectSample', db_sampler=db_sampler),
dict(
    type='GlobalRotScaleTrans',
    rot_range=[-0.3925, 0.3925],
    scale_ratio_range=[0.95, 1.05],
    translation_std=[0, 0, 0]),
dict(
    type='RandomFlip3D',
    sync_2d=False,
    flip_ratio_bev_horizontal=0.5,
    flip_ratio_bev_vertical=0.5),
dict(type='ObjectNameExpansion', classes=class_names, task_names=TASK_NAMES, class_mapping=CLASS_MAPPING),
dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
dict(type='ObjectNameFilter', classes=total_class_names),
dict(type='PointShuffle'),
dict(type='DefaultFormatBundle3D', class_names=total_class_names),
dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

    
test_pipeline = [
    dict(
        type='LoadPointsFromFileFeather',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        shift_height=False,
        use_color=False,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweepsFeather',
        coord_type="LIDAR",
        sweeps_num=5,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False,
        shift_height=False,
        use_color=False,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        # Add double-flip augmentation
        flip=True,
        pcd_horizontal_flip=True,
        pcd_vertical_flip=True,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', sync_2d=False),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFileFeather',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        shift_height=False,
        use_color=False,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweepsFeather',
        coord_type="LIDAR",
        sweeps_num=5,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False,
        shift_height=False,
        use_color=False,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

evaluation = dict(interval=20, pipeline=eval_pipeline)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/av2_infos_train.pkl'.format(VERSION),
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        sampler_type=SAMPLER_TYPE,
        task_names=TASK_NAMES,
        class_mapping=CLASS_MAPPING),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/av2_infos_val.pkl'.format(VERSION),
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/av2_infos_val.pkl'.format(VERSION),
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))


optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)

checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'