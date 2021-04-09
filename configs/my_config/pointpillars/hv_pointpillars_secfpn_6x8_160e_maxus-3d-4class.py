_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_maxus.py',
    '../_base_/datasets/maxus-3d.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=1)
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
total_epochs = 80
