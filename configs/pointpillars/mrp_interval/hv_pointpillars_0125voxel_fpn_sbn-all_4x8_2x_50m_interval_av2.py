import os 

user = os.getlogin()

if user == "nperi":
    data_root = "/ssd0/nperi/Sensor/"
elif user == "ubuntu":
    data_root = "/home/ubuntu/Workspace/Data/Sensor/"


dataset_type = 'AV2Dataset'
VERSION = "av2_mmdet3d_trainval"

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

SAMPLER_TYPE = "STANDARD"

voxel_size = [0.125, 0.125, 6]
point_cloud_range = [-50, -50, -3, 50, 50, 3]
output_shape  = [int((abs(point_cloud_range[0]) + abs(point_cloud_range[3])) / voxel_size[0]), int((abs(point_cloud_range[1]) + abs(point_cloud_range[4])) / voxel_size[1])]

start_point_cloud_range = point_cloud_range
end_point_cloud_range = point_cloud_range

file_client_args = dict(backend='disk')
# For AV2 we usually do 26-class detection
CLASS_NAMES = [
    'REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG'
]

TOTAL_CLASS_NAMES = CLASS_NAMES

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

# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000),
        deterministic=False),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=6,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=output_shape), #
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=len(TOTAL_CLASS_NAMES),
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[point_cloud_range],
            scales=[1, 2, 4],
            sizes=[
                [2.5981, 0.8660, 1.],  # 1.5 / sqrt(3)
                [1.7321, 0.5774, 1.],  # 1 / sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.01,
            min_bbox_size=0,
            max_num=500)))

# If point cloud range is changed, the models should also change their point
# cloud range accordingly

# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))

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
classes=CLASS_NAMES,
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
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilterInterval', start_point_cloud_range=start_point_cloud_range, end_point_cloud_range=end_point_cloud_range),
    dict(type='ObjectRangeFilterInterval', start_point_cloud_range=start_point_cloud_range, end_point_cloud_range=end_point_cloud_range),
    dict(type='ObjectNameFilter', classes=CLASS_NAMES),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=CLASS_NAMES),
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
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilterInterval', start_point_cloud_range=start_point_cloud_range, end_point_cloud_range=end_point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=CLASS_NAMES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
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
        class_names=CLASS_NAMES,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '{}/av2_infos_train.pkl'.format(VERSION),
            pipeline=train_pipeline,
            classes=CLASS_NAMES,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'),
        data_root=data_root,
        ann_file=data_root + '{}/av2_infos_train.pkl'.format(VERSION),
        pipeline=train_pipeline,
        classes=CLASS_NAMES,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/av2_infos_val.pkl'.format(VERSION),
        pipeline=test_pipeline,
        classes=CLASS_NAMES,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/av2_infos_val.pkl'.format(VERSION),
        pipeline=test_pipeline,
        classes=CLASS_NAMES,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=20, pipeline=eval_pipeline)

# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[20, 23])
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)


# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

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