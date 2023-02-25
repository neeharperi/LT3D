_base_ = [
    '../_base_/datasets/anm_nus-mono3d.py', '../_base_/models/pgd_nus.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    bbox_head=dict(
        pred_bbox2d=True,
        group_reg_dims=(2, 1, 3, 1, 2,
                        4),  # offset, depth, size, rot, velo, bbox2d
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            (),  # velo
            (256, )  # bbox2d
        ),
        loss_depth=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        bbox_coder=dict(
            type='PGDBBoxCoder',
            base_depths=((37.152, 24.632), (31.99, 21.124), 
                        (20.606, 13.679), (23.893, 15.209), 
                        (20.571, 14.341), (34.157, 20.107), 
                        (27.457, 15.528), (22.736, 15.011), 
                        (22.193, 16.328), (24.278, 16.049), 
                        (22.348, 13.704), (40.911, 26.341), 
                        (39.687, 23.974), (22.298, 10.944), 
                        (24.985, 12.478), (29.132, 16.155), 
                        (18.995, 12.011), (29.624, 21.013)),
                         
            base_dims=( (4.62, 1.73, 1.96), (6.93, 2.83, 2.51),
                        (12.56, 3.89, 2.94), (11.22, 3.5, 2.95),
                        (6.68, 3.21, 2.85), (1.7, 1.28, 0.6),
                        (2.11, 1.46, 0.78), (5.04, 1.85, 2.03),
                        (0.73, 1.77, 0.67), (0.53, 1.38, 0.51),
                        (0.69, 1.83, 0.73), (0.71, 1.74, 0.72),
                        (0.95, 1.17, 0.63), (1.18, 1.71, 0.62),
                        (0.67, 1.06, 0.6), (1.08, 1.26, 1.01),
                        (0.41, 1.07, 0.41), (0.5, 0.98, 2.53)),
            code_size=9)),
    # set weight 1.0 for base 7 dims (offset, depth, size, rot)
    # 0.05 for 2-dim velocity and 0.2 for 4-dim 2D distance targets
    train_cfg=dict(code_weight=[
        1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2
    ]),
    test_cfg=dict(nms_pre=1000, nms_thr=0.8, score_thr=0.01, max_per_img=200))

class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 
                'emergency_vehicle', 'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 
                'personal_mobility', 'pushable_pullable', 'debris', 'traffic_cone', 'barrier']
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
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.004, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
total_epochs = 24
evaluation = dict(interval=1)
runner = dict(max_epochs=total_epochs)
