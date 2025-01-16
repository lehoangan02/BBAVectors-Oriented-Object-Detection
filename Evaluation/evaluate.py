import numpy as np
from mmeval import DOTAMeanAP
from mmeval import Accuracy
import argparse

result_path = 'Task1_bridge.txt'
truth_path = 'annoBridge.txt'

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the result')
    parser.add_argument('--result_path', type=str, default='Task1_bridge.txt', help='Path to the result file')
    parser.add_argument('--truth_path', type=str, default='annoBridge.txt', help='Path to the truth file')
    parser.add_argument('--save_path', type=str, default='result.txt', help='Path to save the mAP result')
    args = parser.parse_args()
    return args

def parse_result(line):
    parts = line.split(' ')
    image_id = parts[0]
    confidence = float(parts[1])
    bbox = np.array([float(x) for x in parts[2:]])
    return image_id, confidence, bbox
def read_result(result_path):
    with open(result_path, 'r') as f:
        lines = f.readlines()
    image_id_list = []
    confidence_list = []
    bbox_list = []
    for line in lines:
        image_id, confidence, bbox = parse_result(line)
        image_id_list.append(image_id)
        confidence_list.append(confidence)
        bbox_list.append(bbox)
    return image_id_list, confidence_list, bbox_list
def parse_truth(line):
    parts = line.split(' ')
    bbox = np.array([float(x) for x in parts[:8]])
    class_string = parts[8]
    difficulty = parts[9]
    return bbox, class_string, difficulty
        
def read_truth(truth_path):
    with open(truth_path, 'r') as f:
        lines = f.readlines()
    image_id_list = []
    bbox_list = []
    label_list = []
    for line in lines:
        bbox, class_string, difficulty = parse_truth(line)
        image_id_list.append(class_string)
        bbox_list.append(bbox)
        label_list.append(0)
    return image_id_list, bbox_list, label_list

if __name__ == '__main__':

    args = parse_args()
    image_id_list, confidence_list, bbox_list = read_result(result_path)
    image_id_list_truth, bbox_list_truth, label_list_truth = read_truth(truth_path)

    prediction = {
        'bboxes': np.stack(bbox_list, axis=0),
        'scores': np.ones(len(confidence_list)),
        'labels': np.zeros(len(confidence_list))
    }
    groundtruth = {
        'bboxes': np.stack(bbox_list_truth, axis=0),
        'labels': np.zeros(len(label_list_truth)),
        'bboxes_ignore': np.zeros((0, 8)),
        'labels_ignore': np.zeros((0, ))
    }

    accuracy = DOTAMeanAP(num_classes=1)
    result = accuracy(predictions=[prediction, ], groundtruths=[groundtruth, ])
    print(result)
    with open(args.save_path, 'w') as f:
        f.write(str(result))
    print('Result saved to', args.save_path)
    print('Done')

