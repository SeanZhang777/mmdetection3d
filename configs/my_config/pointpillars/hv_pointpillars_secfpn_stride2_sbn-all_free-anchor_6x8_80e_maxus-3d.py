_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_stride2_maxus.py',
    '../_base_/datasets/maxus-3d.py', '../_base_/schedules/cyclic_80e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pts_bbox_head=dict(
        _delete_=True,
        type='FreeAnchor3DHead',
        num_classes=4,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-60.0, -32.0, 0.0, 60.0, 32.0, 0.0],
                    [-60.0, -32.0, 0.0, 60.0, 32.0, 0.0],
                    [-60.0, -32.0, 0.0, 60.0, 32.0, 0.0],
                    [-60.0, -32.0, 0.0, 60.0, 32.0, 0.0]],
            sizes=[
                [2.083, 4.846, 1.794],  # Car
                [3.050, 13.296, 3.237], # Large_Vehicle
                [0.718, 0.828, 1.744],  # Pedestrian
                [0.903, 1.950, 1.637]  # Cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)))
        
