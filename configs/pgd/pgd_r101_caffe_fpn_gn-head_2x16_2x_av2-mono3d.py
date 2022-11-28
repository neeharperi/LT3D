import os 

user = os.getlogin()

if user == "nperi":
    data_root = "/ssd0/nperi/Sensor/"
elif user == "ubuntu":
    data_root = "/home/ubuntu/Workspace/Data/Sensor/"

dataset_type = 'AV2MonoDataset'
VERSION = "av2_mmdet3d_trainval"

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

class_names = [
    'REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG'
]

model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe'),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='PGDHead',
        num_classes=len(class_names),
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=True,
        pred_velo=True,
        pred_bbox2d=True,
        pred_keypoints=False,
        dir_offset=0.7854,  # pi/4
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2, 4),  # offset, depth, size, rot, velo, bbox2d
        cls_branch=(256, ),
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            (),  # velo
            (256, )  # bbox2d
        ),
        dir_branch=(256, ),
        attr_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_depth=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=True,
        use_depth_classifier=True,
        depth_branch=(256, ),
        depth_range=(0, 50),
        depth_unit=10,
        division='uniform',
        depth_bins=6,
        bbox_coder=dict(
            type='PGDBBoxCoder',
            base_depths=[(67.916, 49.937), (46.656, 33.858), (56.606, 40.822), (38.339, 28.203), 
                        (76.658, 50.601), (54.712, 40.125), (73.337, 48.471), (37.717, 30.003), 
                        (55.527, 41.531), (37.9, 28.919), (63.728, 45.957), (56.305, 41.804), 
                        (56.094, 42.341), (70.247, 51.26), (35.152, 24.898), (44.353, 30.747), 
                        (42.142, 32.147), (48.5, 36.594), (39.861, 28.927), (29.193, 19.689), 
                        (61.041, 46.659), (62.564, 47.949), (50.827, 39.056), (43.389, 31.502), 
                        (71.797, 52.945), (97.331, 50.898)]
            base_dims=[(4.48, 1.94, 1.73), (0.7, 0.77, 1.76),(1.12, 0.81, 1.77),(1.26, 0.85, 1.63),
                        (0.82, 0.73, 1.81),(0.36, 0.31, 1.06), (0.36, 0.33, 0.88), (0.44, 1.51, 2.6),
                        (0.69, 0.66, 1.09), (0.36, 0.98, 3.09), (0.32, 0.99, 1.42), (6.66, 2.68, 3.05),
                        (11.67, 2.97, 3.3), (7.7, 2.81, 3.5), (9.79, 2.85, 3.35), (7.42, 2.87, 3.26),
                        (7.64, 3.32, 3.65), (8.93, 2.79, 3.1), (10.54, 2.94, 3.29), (3.25, 3.2, 3.75),
                        (1.65, 0.62, 1.23), (1.86, 0.73, 1.34), (1.26, 0.6, 1.38), (0.98, 0.76, 1.15),
                        (0.87, 0.65, 1.2), (1.0, 0.45, 0.8)],
            code_size=9)),
    # set weight 1.0 for base 7 dims (offset, depth, size, rot)
    # 0.05 for 2-dim velocity and 0.2 for 4-dim 2D distance targets
    train_cfg=dict(allowed_border=0, 
    pos_weight=-1,
    code_weight=[
        1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2
    ]),
    test_cfg=dict(use_rotate_nms=True, nms_across_levels=False, nms_pre=1000, nms_thr=0.8, score_thr=0.01, min_bbox_size=0, max_per_img=200))

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/av2_infos_train_mono3d.coco.json'.format(VERSION),
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='Camera'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/av2_infos_val_mono3d.coco.json'.format(VERSION),
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '{}/av2_infos_val_mono3d.coco.json'.format(VERSION),
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'))
evaluation = dict(interval=24)

# optimizer
optimizer = dict(lr=0.004, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
total_epochs = 24
runner = dict(max_epochs=total_epochs)

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

