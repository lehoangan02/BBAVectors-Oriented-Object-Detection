#!/bin/bash

# Things you need to change:
# 1. data_dir: Path to the directory containing the validation data
# 2. model_dir: Path to the directory containing the model weights

# How to run:
# Make sure you are using Git bash if you are on Windows
# cd Custom/Scripts
# chmod +x evaluate_models.sh
# ./evaluate_models.sh
echo "Running evaluation for all models in the model directory..."
# Define variables
data_dir="./datasets/Validate_DOTA_1_0.5"
# data_dir="./datasets/MiniTrainV1.1" # testing
# Directory structure:
# -|datasets
# ---|your_validation_data
# -----|images
# -----|labelTxt
conf_thresh=0.1
batch_size=16
dataset="dota"
phase="eval"
model_dir="c0e1"
# Directory structure:
# -|weights_dota
# ---| your_model_weights
# -----| model_5.pth
# -----| model_10.pth
# -----| ...
eval_script="dota_evaluation_task1.py"
eval_dir="datasets/DOTA_devkit"
result_dir="Result/c0e1"
# A folder named "Result" will be created in the current directory to store the evaluation results

# Array of model epochs to evaluate
epochs=(6 7 8 9 10)
cd ../..

# Create the result directory if it doesn't exist
mkdir -p "Result"
mkdir -p "$result_dir"

# Loop through the epochs and run the evaluation
for epoch in "${epochs[@]}"; do
    model_path="${model_dir}/model_${epoch}.pth"
    echo "Running evaluation for model at epoch ${epoch}..."
    python main.py --data_dir "$data_dir" --conf_thresh "$conf_thresh" --batch_size "$batch_size" --dataset "$dataset" --phase "$phase" --resume "$model_path"
    
    # Change directory to evaluation script location and run evaluation
    echo "Running DOTA evaluation for model at epoch ${epoch}..."
    (cd "$eval_dir" && python "$eval_script") | tee "$result_dir/evaluation_result_for_epoch_${epoch}.txt"
done
