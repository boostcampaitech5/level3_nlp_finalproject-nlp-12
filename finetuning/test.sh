#!/bin/bash

config=("config.yaml")

# Check if output_logs directory exists, if not, create it
if [ ! -d "output_logs" ]; then
  mkdir output_logs
fi

echo "Start finetuning with ${config}..."
nohup python3 train.py "${config}" > output_logs/output.txt 2>&1 &
wait $!
echo "Finetuning with ${config} has been completed!"

echo "Start merging finetuned model with original model..."
nohup python3 merge_lora.py "${config}" >> output_logs/output.txt 2>&1 &
wait $!
echo "Merging has been completed!"

echo "Start testing model with several samples..."
nohup python3 inference.py "${config}" > output_logs/inference.txt 2>&1 &
echo "Testing has been completed!"