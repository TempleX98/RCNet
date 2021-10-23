_base_ = [
    '../retinanet/retinanet_r50_fpn_1x_coco.py'
]
# optimizer
model = dict(
    neck=dict(type='RCFPN', norm_cfg=dict(type='BN', requires_grad=True)))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=4, # bs=4 when using BN
    workers_per_gpu=2)