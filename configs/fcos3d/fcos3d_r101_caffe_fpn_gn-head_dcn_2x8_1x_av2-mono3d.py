
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
    type='FCOSMono3D',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSMono3DHead',
        num_classes=26,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=True,
        pred_velo=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2),  # offset, depth, size, rot, velo
        cls_branch=(256, ),
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            ()  # velo
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
        bbox_coder=dict(type='FCOS3DBBoxCoder', code_size=9),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=True),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.05,
        score_thr=0.01,
        min_bbox_size=0,
        max_per_img=200))

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
evaluation = dict(interval=12)

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
total_epochs = 12

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