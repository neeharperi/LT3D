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

wide = True
wide=512

use_sampler = True
sampler_type = "standard"

voxel_size = [0.2, 0.2, 8]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
sparse_shape = [int((abs(point_cloud_range[2]) + abs(point_cloud_range[5])) / voxel_size[2]) + 1, int((abs(point_cloud_range[1]) + abs(point_cloud_range[4])) / voxel_size[1]), int((abs(point_cloud_range[0]) + abs(point_cloud_range[3])) / voxel_size[0])]
grid_size = [int((abs(point_cloud_range[0]) + abs(point_cloud_range[3])) / voxel_size[0]), int((abs(point_cloud_range[1]) + abs(point_cloud_range[4])) / voxel_size[1]), int((abs(point_cloud_range[2]) + abs(point_cloud_range[5])) / voxel_size[2])]

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
                                ['pushable_pullable'], ['debris'], ['traffic_cone'], ['barrier']],
                 "group" : [['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle'], 
                            ['adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility'], 
                            ['pushable_pullable', 'debris', 'traffic_cone', 'barrier']],
                 "all" : [['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                            'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                            'pushable_pullable', 'debris', 'traffic_cone', 'barrier']]
                 }

nms = ["W", "W", "W", "W", "W", "W", "W", "W", "W", "W", 
       "W", "W", "W", "W", "W", "W", "W", "W", "X", "X", "X", "X"]
#nms = ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", 
#       "A", "A", "A", "A", "A", "A", "A", "X", "X", "X", "X"]

hierarchical_softmax = [['car', 'vehicle', 'object'], ['truck', 'vehicle', 'object'], ['trailer', 'vehicle', 'object'], ['bus', 'vehicle', 'object'], ['construction_vehicle', 'vehicle', 'object'], ['bicycle', 'vehicle', 'object'], ['motorcycle', 'vehicle', 'object'], ['emergency_vehicle', 'vehicle', 'object'], 
                        ['adult', 'pedestrian', 'object'], ['child', 'pedestrian', 'object'], ['police_officer', 'pedestrian', 'object'], ['construction_worker', 'pedestrian', 'object'], ['stroller', 'pedestrian', 'object'], ['personal_mobility', 'pedestrian', 'object'], 
                        ['pushable_pullable', 'movable', 'object'], ['debris', 'movable', 'object'], ['traffic_cone', 'movable', 'object'], ['barrier', 'movable', 'object'],
                        ["vehicle"], ["pedestrian"], ["movable"], ["object"]]

hierarchical_softmax = [[total_class_names.index(g) for g in hs] for hs in hierarchical_softmax]
hierarchy = {"TRAIN" : False,
             "TEST" : False,
             "GROUP" : hierarchical_softmax}

model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=20, voxel_size=voxel_size, max_voxels=(30000, 40000), point_cloud_range=point_cloud_range, deterministic=False),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(512, 512)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[dict(num_class=22, class_names=total_class_names)],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=wide,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=point_cloud_range,
            max_num=4096,
            score_threshold=0.01,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9),
        separate_head=dict(
            type='DCNSeparateHead',
            dcn_config=dict(
                type='DCN',
                in_channels=wide,
                out_channels=wide,
                kernel_size=3,
                padding=1,
                groups=4),
            head_conv=wide,
            init_bias=-2.19,
            final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            hierarchy=hierarchy)),
    test_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            post_center_range=point_cloud_range,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.01,
            out_size_factor=4,
            nms_type='rotate',
            use_rotate_nms=True,
            max_num=500,
            pre_max_size=1000,
            post_max_size=500,
            nms_thr=0.2,
            wide=wide,
            nms=nms,
            hierarchy=hierarchy)))

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
        use_dim=[0, 1, 2, 3, 4],
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
            use_dim=[0, 1, 2, 3, 4],
            file_client_args=file_client_args,
            pad_empty_sweeps=True,
            remove_close=True),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='ObjectNameFilter', classes=class_names),
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
        dict(type='ObjectNameExpansion', classes=class_names, task_names=task_names, class_mapping=class_mapping),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectNameFilter', classes=total_class_names),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=total_class_names),
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
            use_dim=[0, 1, 2, 3, 4],
            file_client_args=file_client_args,
            pad_empty_sweeps=True,
            remove_close=True),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='ObjectNameFilter', classes=class_names),
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
        dict(type='ObjectNameExpansion', classes=class_names, task_names=task_names, class_mapping=class_mapping),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectNameFilter', classes=total_class_names),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=total_class_names),
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
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=[0.95, 1.0, 1.05],
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

if use_sampler:
    train_data=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '{}/nuscenes_infos_val.pkl'.format(VERSION),
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'),
        data_root=data_root,
        ann_file=data_root + '{}/nuscenes_infos_val.pkl'.format(VERSION),
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
        ann_file=data_root + '{}/nuscenes_infos_val.pkl'.format(VERSION),
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
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

evaluation = dict(interval=20, pipeline=eval_pipeline)

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

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

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
