_base_ = './hv_pointpillars_fpn_sbn-all_free-anchor_6x8_80e_maxus-3d.py'

model = dict(
    pretrained=dict(pts='open-mmlab://regnetx_400mf'),
    pts_backbone=dict(
        _delete_=True,
        type='NoStemRegNet',
        arch='regnetx_400mf',
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        strides=(1, 2, 2, 2),
        base_channels=64,
        stem_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        norm_eval=False,
        style='pytorch'),
    pts_neck=dict(in_channels=[64, 160, 384]))