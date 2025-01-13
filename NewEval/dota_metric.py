import numpy as np
from mmeval import DOTAMetric
num_classes = 15
dota_metric = DOTAMetric(num_classes=15)
def _gen_bboxes(num_bboxes, img_w=256, img_h=256):
    # random generate bounding boxes in 'xywha' formart.
    x = np.random.rand(num_bboxes, ) * img_w
    y = np.random.rand(num_bboxes, ) * img_h
    w = np.random.rand(num_bboxes, ) * (img_w - x)
    h = np.random.rand(num_bboxes, ) * (img_h - y)
    a = np.random.rand(num_bboxes, ) * np.pi / 2
    return np.stack([x, y, w, h, a], axis=1)
prediction = {
    'bboxes': _gen_bboxes(10),
    'scores': np.random.rand(10, ),
    'labels': np.random.randint(0, num_classes, size=(10, ))
}
groundtruth = {
    'bboxes': _gen_bboxes(10),
    'labels': np.random.randint(0, num_classes, size=(10, )),
    'bboxes_ignore': _gen_bboxes(5),
    'labels_ignore': np.random.randint(0, num_classes, size=(5, ))
}
dota_metric(predictions=[prediction, ], groundtruths=[groundtruth, ])  
{'mAP@0.5': ..., 'mAP': ...}