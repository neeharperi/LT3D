_base_ = ['./centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_av2.py']

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -3.0, 54, 54, 3.0]
# For AV2 we usually do 26-class detection
class_names = [
    'REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG'
]

model = dict(
    pts_voxel_layer=dict(
        voxel_size=voxel_size, point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(sparse_shape=[31, 1440, 1440]),
    pts_bbox_head=dict(
        bbox_coder=dict(
            voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2])),
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 30],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(grid_size=[1440, 1440, 30], voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2])))

dataset_type = 'AV2Dataset'
data_root = '/ssd0/nperi/Sensor/'
VERSION = "av2_mmdet3d_trainval"
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + '{}/av2_dbinfos_train.pkl'.format(VERSION),
    rate=1.0,
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
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))