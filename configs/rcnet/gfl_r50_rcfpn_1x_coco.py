_base_ = [
    '../gfl/gfl_r50_fpn_1x_coco.py'
]
# optimizer
model = dict(
    neck=dict(
        type='RCFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='BN', requires_grad=True)))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=4, # bs=4 when using BN
    workers_per_gpu=2)