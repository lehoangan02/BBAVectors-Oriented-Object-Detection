import os
import evaluate
import argparse

dummy_image_filename = 'x-10'
def parse_args():
    parser = argparse.ArgumentParser(description='Generate perfect result')
    parser.add_argument('--label_filepath', type=str, default='annoBridge.txt', help='Path to the label file')
    parser.add_argument('--result_filepath', type=str, default='Task1_bridge.txt', help='Path to the result file')
    args = parser.parse_args()
    return args
def generatePerfectResult(label_filepath, result_filepath):
    image_id_list, bbox_list, label_list = evaluate.read_truth(label_filepath)
    with open(result_filepath, 'w') as f:
        for i in range(len(bbox_list)):
            f.write(f"{dummy_image_filename} 1 {bbox_list[i][0]} {bbox_list[i][1]} {bbox_list[i][2]} {bbox_list[i][3]} {bbox_list[i][4]} {bbox_list[i][5]} {bbox_list[i][6]} {bbox_list[i][7]}\n")

if __name__ == '__main__':
    args = parse_args()
    generatePerfectResult(args.label_filepath, args.result_filepath)
    print("Generated perfect result at", args.result_filepath)