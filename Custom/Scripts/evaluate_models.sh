#!/bin/bash

# Define variables
data_dir="./datasets/MiniTrainV1.1"
conf_thresh=0.1
batch_size=16
dataset="dota"
phase="eval"
model_dir="75_epoch_resnet152_attempt"
eval_script="dota_evaluation_task1.py"
eval_dir="datasets/DOTA_devkit"
result_dir="Result"

# Create the result directory if it doesn't exist
mkdir -p "$result_dir"

# Array of model epochs to evaluate
epochs=(60 65 70 75)
cd ../..
# Loop through the epochs and run the evaluation
for epoch in "${epochs[@]}"; do
    model_path="${model_dir}/model_${epoch}.pth"
    echo "Running evaluation for model at epoch ${epoch}..."
    python main.py --data_dir "$data_dir" --conf_thresh "$conf_thresh" --batch_size "$batch_size" --dataset "$dataset" --phase "$phase" --resume "$model_path"
    
    # Change directory to evaluation script location and run evaluation
    echo "Running DOTA evaluation for model at epoch ${epoch}..."
    (cd "$eval_dir" && python "$eval_script") | tee "$result_dir/eval_epoch_${epoch}.txt"
done
