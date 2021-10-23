
RCNet: Reverse Feature Pyramid and Cross-scale Shift Network for Object Detection (ACM MM'21)
---------------------
By Zhuofan Zong, Qianggang Cao, Biao Leng


Introduction
----------------
Feature pyramid networks (FPN) are widely exploited for multi-scale feature fusion in existing advanced object detection frameworks. Numerous previous works have developed various structures for bidirectional feature fusion, all of which are shown to improve the detection performance effectively. We observe that these complicated network structures require feature pyramids to be stacked in a fixed order, which introduces longer pipelines and reduces the inference speed. Moreover, semantics from non-adjacent levels are diluted in the feature pyramid since only features at adjacent pyramid levels are merged by the local fusion operation in a sequence manner. To address these issues, we propose a novel architecture named RCNet, which consists of Reverse Feature Pyramid (RevFP) and Cross-scale Shift Network (CSN). RevFP utilizes local bidirectional feature fusion to simplify the bidirectional pyramid inference pipeline. CSN directly propagates representations to both adjacent and non-adjacent levels to enable multi-scale features more correlative. Extensive experiments on the MS COCO dataset demonstrate RCNet can consistently bring significant improvements over both one-stage and two-stage detectors with subtle extra computational overhead. In particular, RetinaNet is boosted to 40.2 AP, which is 3.7 points higher than baseline, by replacing FPN with our proposed model. On COCO test-dev, RCNet can achieve very competitive performance with a single-model single-scale 50.5 AP.

Models
---------------

Pretrained models will be available.

Training and Testing
--------------
This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please follow mmdetection on how to install and use this repo.

Results on MS COCO
---------
| Detector | Backbone | Neck | Lr schd | mAP(val) | mAP(test)|
|----------|--------|------|-----|-----------|----|
| RetinaNet | R50 | RCNet | 1x | 40.2 | - |
| ATSS | R50 | RCNet | 1x | 42.6 | - |
| GFL | R50 | RCNet | 1x | 43.1 | - |
| GFL | R101 | RCNet | 2x | 47.1 | 47.4 |
| GFL | X101-64x4d | RCNet | 2x | 48.9 | 49.2 |
| GFL | X101-64x4d-DCN | RCNet | 2x | 50.2 | 50.5 |


Citations
------------

If you find RCNet useful in your research, please consider citing:
```
@inproceedings{zong2021rcnet,
author = {Zong, Zhuofan and Cao, Qianggang and Leng, Biao},
title = {RCNet: Reverse Feature Pyramid and Cross-scale Shift Network for Object Detection},
booktitle = {ACM MM},
pages = {5637–5645},
year = {2021}
}
```

License
--------
This project is released under the [Apache 2.0 license](LICENSE)
