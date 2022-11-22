import os 

user = os.getlogin()

if user == "nperi":
    data_root = '/ssd0/nperi/nuScenes/'
elif user == "ubuntu":
    data_root = "/home/ubuntu/Workspace/Data/nuScenes/"

dataset_type = 'NuScenesDataset'
VERSION = "nusc_mmdet3d_trainval"
file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

use_sampler = True
sampler_type = "standard"

voxel_size = [0.25, 0.25, 8]
point_cloud_range = [-50, -50, -5, 50, 50, 3]
output_shape  = [int((abs(point_cloud_range[0]) + abs(point_cloud_range[3])) / voxel_size[0]), int((abs(point_cloud_range[1]) + abs(point_cloud_range[4])) / voxel_size[1])]

file_client_args = dict(backend='disk')
class_names = [
'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
'pushable_pullable', 'debris', 'traffic_cone', 'barrier'
]

total_class_names = class_names + ["vehicle", "pedestrian", "movable", "object"]

task_names = {"standard": ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                            'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                            'pushable_pullable', 'debris', 'traffic_cone', 'barrier'],
              "group": ["vehicle", "pedestrian", "movable"],
              "all" : ["object"]
             }

class_mapping = {"standard": [['car'], ['truck'], ['trailer'], ['bus'], ['construction_vehicle'], ['bicycle'], ['motorcycle'], ['emergency_vehicle'], 
                                ['adult'], ['child'], ['police_officer'], ['construction_worker'], ['stroller'], ['personal_mobility'], 
                                ['pushable_pullable'], ['debris'], ['traffic_cone'], ['barrier'] ],
                 "group" : [['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle'], 
                            ['adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility'], 
                            ['pushable_pullable', 'debris', 'traffic_cone', 'barrier']],
                 "all" : [['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                            'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                            'pushable_pullable', 'debris', 'traffic_cone', 'barrier']]
                 }

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.

# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))

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
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=output_shape),
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
        num_classes=len(total_class_names),
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

db_sampler = dict(
data_root=data_root,
info_path=data_root + '{}/nuscenes_dbinfos_train.pkl'.format(VERSION),
rate=1.0,
sampler_type=sampler_type, 
task_names=task_names,
class_mapping=class_mapping,
prepare=dict(
    filter_by_difficulty=[-1],
    filter_by_min_points=dict(
        car=5,
        truck=5,
        construction_vehicle=5,
        bus=5,
        trailer=5,
        emergency_vehicle=5,
        motorcycle=5,
        bicycle=5,
        adult=5,
        child=5,
        police_officer=5,
        construction_worker=5,
        personal_mobility=5,
        stroller=5,
        pushable_pullable=5,
        barrier=5,
        traffic_cone=5,
        debris=5,
        )),
classes=class_names,
sample_groups=dict(
    car=2,
    truck=3,
    construction_vehicle=7,
    bus=4,
    trailer=6,
    emergency_vehicle=8,
    motorcycle=6,
    bicycle=6,
    adult=2,
    child=8,
    police_officer=8,
    construction_worker=6,
    personal_mobility=8,
    stroller=8,
    pushable_pullable=6,
    barrier=2,
    traffic_cone=2,
    debris=8,
    ),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3],
        file_client_args=file_client_args))

if use_sampler: 
    train_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=file_client_args),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=9,
            file_client_args=file_client_args),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='ObjectNameFilter', classes=class_names),
        dict(type='ObjectSample', db_sampler=db_sampler),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectNameFilter', classes=class_names),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
else:
    train_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=file_client_args),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=9,
            file_client_args=file_client_args),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectNameFilter', classes=class_names),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

if use_sampler:
    train_data=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '{}/nuscenes_infos_train.pkl'.format(VERSION),
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'),
        data_root=data_root,
        ann_file=data_root + '{}/nuscenes_infos_train.pkl'.format(VERSION),
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        sampler_type=sampler_type,
        task_names=task_names,
        class_mapping=class_mapping)
else:
    train_data=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/nuscenes_infos_train.pkl'.format(VERSION),
        pipeline=train_pipeline,
        classes=CLASS_NAMES,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        sampler_type=sampler_type,
        task_names=task_names,
        class_mapping=class_mapping),
    
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=train_data,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/nuscenes_infos_val.pkl'.format(VERSION),
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/nuscenes_infos_val.pkl'.format(VERSION),
        pipeline=test_pipeline,
        classes=class_names,
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