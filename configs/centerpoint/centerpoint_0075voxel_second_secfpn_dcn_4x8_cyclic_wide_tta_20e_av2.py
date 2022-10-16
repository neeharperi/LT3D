_base_ = './centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_20e_av2.py'

WIDE = True
WIDE_DIM=512

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -3, 54, 54, 3]

file_client_args = dict(backend='disk')
class_names = [
    'REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG'
]

model = dict(
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[dict(num_class=26, class_names=['REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER', 'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 
                                                'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER', 
                                                'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG']),],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=WIDE_DIM,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-54, -54, -3, 54, 54, 3],
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
            grid_size=[1440, 1440, 30],
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
            pre_max_size=1000,
            nms_thr=0.2,
            wide=WIDE)))


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


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))
