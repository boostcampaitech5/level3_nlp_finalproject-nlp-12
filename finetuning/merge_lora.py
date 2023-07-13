import sys
import yaml

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.arguments import TrainArguments


def apply_lora(train_args: TrainArguments):
    print(f"Loading the base model from {train_args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(train_args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    base_tokenizer = AutoTokenizer.from_pretrained(train_args.base_model, use_fast=False)

    print(f"Loading the LoRA adapter from {train_args.finetune_dir}")
    lora_model = PeftModel.from_pretrained(base, train_args.finetune_dir,torch_dtype=torch.float16)

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {train_args.merge_dir}")
    model.save_pretrained(train_args.merge_dir)
    base_tokenizer.save_pretrained(train_args.merge_dir)

    
def load_config(config_file: str):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return TrainArguments(**config)


def main():
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = './config.yaml'
    train_args = load_config(config_path)
    apply_lora(train_args)


if __name__ == "__main__":
    main()