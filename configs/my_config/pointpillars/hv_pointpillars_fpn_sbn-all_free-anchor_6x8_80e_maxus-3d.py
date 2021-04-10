_base_ = [
    '../_base_/models/hv_pointpillars_fpn_maxus.py',
    '../_base_/datasets/maxus-3d.py', '../_base_/schedules/cyclic_80e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pts_bbox_head=dict(
        _delete_=True,
        type='FreeAnchor3DHead',
        num_classes=4,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-60.0, -32.0, 0.0, 60.0, 32.0, 0.0]],
            scales=[1, 1, 1],
            sizes=[
		[2.083, 4.846, 1.794],  # Car
                [3.050, 13.296, 3.237], # Large_Vehicle
                [0.718, 0.828, 1.744],  # Pedestrian
                [0.903, 1.950, 1.637],  # Cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=True))),
        assigner_per_size=False))
        
