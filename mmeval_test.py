import numpy as np
from mmeval import DOTAMeanAP
from mmeval import Accuracy
num_classes = 15
dota_metric = DOTAMeanAP(num_classes=15)
# def _gen_bboxes(num_bboxes, img_w=256, img_h=256):
#     # random generate bounding boxes in 'xywha' formart.
#     x = np.random.rand(num_bboxes, ) * img_w
#     y = np.random.rand(num_bboxes, ) * img_h
#     w = np.random.rand(num_bboxes, ) * (img_w - x)
#     h = np.random.rand(num_bboxes, ) * (img_h - y)
#     a = np.random.rand(num_bboxes, ) * np.pi / 2
#     return np.stack([x, y, w, h, a], axis=1)
# def genDefaultBox(num_boxes):
#     x1 = np.full((num_boxes, ), 0)
#     y1 = np.full((num_boxes, ), 0)
#     x2 = np.full((num_boxes, ), 1)
#     y2 = np.full((num_boxes, ), 0)
#     x3 = np.full((num_boxes, ), 1)
#     y3 = np.full((num_boxes, ), 1)
#     x4 = np.full((num_boxes, ), 0)
#     y4 = np.full((num_boxes, ), 1)
#     return np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=1)
# def genDefaultBox2(num_boxes):
#     x1 = np.full((num_boxes, ), 0)
#     y1 = np.full((num_boxes, ), 0)
#     x2 = np.full((num_boxes, ), 1.8)
#     y2 = np.full((num_boxes, ), 0)
#     x3 = np.full((num_boxes, ), 1.9)
#     y3 = np.full((num_boxes, ), 1)
#     x4 = np.full((num_boxes, ), 0)
#     y4 = np.full((num_boxes, ), 1)
#     return np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=1)
# def genDefaultBox3(num_boxes):
#     x1 = np.full((num_boxes, ), 0)
#     y1 = np.full((num_boxes, ), 0)
#     x2 = np.full((num_boxes, ), 1.9)
#     y2 = np.full((num_boxes, ), 0)
#     x3 = np.full((num_boxes, ), 1.8)
#     y3 = np.full((num_boxes, ), 1)
#     x4 = np.full((num_boxes, ), 0)
#     y4 = np.full((num_boxes, ), 1)
#     return np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=1)
# prediction = {
#     'bboxes': _gen_bboxes(10),
#     'scores': np.ones(10, ),
#     'labels': np.random.randint(0, num_classes, size=(10, ))
# }
# groundtruth = {
#     'bboxes': _gen_bboxes(10),
#     'labels': np.random.randint(0, num_classes, size=(10, )),
#     'bboxes_ignore': _gen_bboxes(5),
#     'labels_ignore': np.random.randint(0, num_classes, size=(5, ))
# }
# print(_gen_bboxes(10).shape)
# print(genDefaultBox(10).shape)
# myPrediction = {
#     'bboxes' : genDefaultBox2(1),
#     'scores' : np.ones(1, ),
#     'labels' : np.ones(1, )
# }
# myGroundtruth = {
#     'bboxes' : genDefaultBox3(1),
#     'labels' : np.ones(1, ),
#     'bboxes_ignore' : genDefaultBox(0),
#     'labels_ignore' : np.random.randint(0, num_classes, size=(0, ))
# }
# result = (dota_metric(predictions=[myPrediction, ], groundtruths=[myGroundtruth, ])  )
# print(type(result))



# v-6 0.176608905196 455.2 1970.2 494.6 2016.8 461.6 2037.2 422.2 1990.6
# v-6 0.144105017185 407.6 1913.6 459.4 1982.4 411.8 2011.4 360.0 1942.6
# v-6 0.127993702888 339.4 1915.4 396.0 1993.2 367.6 2010.6 311.0 1932.8
firstPrediction = np.array([455.2, 1970.2, 494.6, 2016.8, 461.6, 2037.2, 422.2, 1990.6])
secondPrediction = np.array([407.6, 1913.6, 459.4, 1982.4, 411.8, 2011.4, 360.0, 1942.6])
thirdPrediction = np.array([339.4, 1915.4, 396.0, 1993.2, 367.6, 2010.6, 311.0, 1932.8])
                             
bboxesPrediction = np.stack([firstPrediction, secondPrediction, thirdPrediction], axis=0)
scores = np.array([0.176608905196, 0.144105017185, 0.127993702888])
labelsPrediction = np.array([0, 0, 0])

# 935.0 1840.0 1029.0 1840.0 1029.0 1860.0 935.0 1860.0 bridge 0
# 967.0 1955.0 1064.0 1955.0 1064.0 2006.0 967.0 2006.0 bridge 0
# 318.0 1940.0 342.0 1923.0 407.0 2016.0 383.0 2033.0 bridge 0
# 387.0 1934.0 411.0 1917.0 476.0 2010.0 452.0 2027.0 bridge 0

firstGroundtruth = np.array([935.0, 1840.0, 1029.0, 1840.0, 1029.0, 1860.0, 935.0, 1860.0])
secondGroundtruth = np.array([967.0, 1955.0, 1064.0, 1955.0, 1064.0, 2006.0, 967.0, 2006.0])
thirdGroundtruth = np.array([318.0, 1940.0, 342.0, 1923.0, 407.0, 2016.0, 383.0, 2033.0])
fourthGroundtruth = np.array([387.0, 1934.0, 411.0, 1917.0, 476.0, 2010.0, 452.0, 2027.0])

bboxes = np.stack([firstGroundtruth, secondGroundtruth, thirdGroundtruth, fourthGroundtruth], axis=0)
labels = np.array([0, 0, 0, 0])

myBriddgePrediction = {
    'bboxes' : bboxesPrediction,
    'scores' : scores,
    'labels' : labelsPrediction
}
myBridgeGroundtruth = {
    'bboxes' : bboxes,
    'labels' : labels,
    'bboxes_ignore' : np.empty((0, 8)),
    'labels_ignore' : np.empty((0, ))
}
result = (dota_metric(predictions=[myBriddgePrediction, ], groundtruths=[myBridgeGroundtruth, ])  )
print(result)


