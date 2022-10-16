_base_ = './centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py'

dataset_type = 'NuScenesDataset'
data_root = '/ssd0/nperi/nuScenes/'
VERSION = "nusc_mmdet3d_trainval"
file_client_args = dict(backend='disk')

WIDE = True
WIDE_DIM=512

SAMPLER_TYPE = "standard"

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
file_client_args = dict(backend='disk')
class_names = [
'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
'pushable_pullable', 'debris', 'traffic_cone', 'barrier'
]

total_class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                    'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                    'pushable_pullable', 'debris', 'traffic_cone', 'barrier', "vehicle", "pedestrian", "movable", "object"]

TASK_NAMES = {"standard": ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                            'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                            'pushable_pullable', 'debris', 'traffic_cone', 'barrier'],
              "group": ["vehicle", "pedestrian", "movable"],
              "all" : ["object"]
             }

CLASS_MAPPING = {"standard": [['car'], ['truck'], ['trailer'], ['bus'], ['construction_vehicle'], ['bicycle'], ['motorcycle'], ['emergency_vehicle'], 
                                ['adult'], ['child'], ['police_officer'], ['construction_worker'], ['stroller'], ['personal_mobility'], 
                                ['pushable_pullable'], ['debris'], ['traffic_cone'], ['barrier'] ],
                 "group" : [['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle'], 
                            ['adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility'], 
                            ['pushable_pullable', 'debris', 'traffic_cone', 'barrier']],
                 "all" : [['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                            'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                            'pushable_pullable', 'debris', 'traffic_cone', 'barrier']]
                 }

NMS = ["W", "W", "W", "W", "W", "W", "W", "W", "W", "W", 
       "W", "W", "W", "W", "W", "W", "W", "X", "X", "X", "X"]
#NMS = ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", 
#       "A", "A", "A", "A", "A", "A", "A", "X", "X", "X", "X"]

GROUP = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                    'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                    'pushable_pullable', 'debris', 'traffic_cone', 'barrier', "vehicle", "pedestrian", "movable", "object"]
#GROUP = ['vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle',
#         'pedestrian', 'pedestrian', 'pedestrian', 'pedestrian', 'pedestrian', 'pedestrian', 
#         'movable', 'movable', 'movable', 'movable', "vehicle", "pedestrian", "movable", "object"]
GROUP = [total_class_names.index(g) for g in GROUP]

model = dict(
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[dict(num_class=22, class_names=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                                                'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                                                'pushable_pullable', 'debris', 'traffic_cone', 'barrier', "vehicle", "pedestrian", "movable", "object"])],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=WIDE_DIM,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-54, -54, -5, 54, 54, 3],
            max_num=16384,
            score_threshold=0.01,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
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
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
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
            use_rotate_nms=True,
            max_num=500,
            pre_max_size=1000,
            nms_thr=0.2,
            wide=WIDE,
            nms=NMS, 
            group=GROUP)))

db_sampler = dict(
data_root=data_root,
info_path=data_root + '{}/nuscenes_dbinfos_train.pkl'.format(VERSION),
rate=1.0,
sampler_type=SAMPLER_TYPE, 
task_names=TASK_NAMES,
class_mapping=CLASS_MAPPING,
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
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
            box_type_3d='LiDAR',
            sampler_type=SAMPLER_TYPE,
            task_names=TASK_NAMES,
            class_mapping=CLASS_MAPPING)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))
