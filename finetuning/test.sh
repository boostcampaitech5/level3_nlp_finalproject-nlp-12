#!/bin/bash

config=("config.yaml")

echo "Start finetuning with ${config}..."
nohup python3 train.py "${config}" > output_logs/output.txt 2>&1 &
wait $!
echo "Finetuning with ${config} has been completed!"

echo "Start merging finetuned model with original model..."
nohup ./merge_lora.sh >> output_logs/output.txt 2>&1 &
wait $!
echo "Merging has been completed!"

echo "Start testing model with several samples..."
nohup python3 inference.py >> output_logs/inference.txt 2>&1 &
echo "Testing has been completed!"