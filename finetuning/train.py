import os
import sys

import torch
import transformers
import yaml

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, BitsAndBytesConfig
from utils.data_loader import *
from utils.arguments import TrainArguments
from utils.compute_metrics import *


def train(train_args: TrainArguments):
    params = train_args.dict()
    print(
        "Training Alpaca-LoRA model with params:\n" +
        "\n".join([f"    {k}: {v}" for k, v in params.items()])
    )
    assert train_args.base_model, "Please specify a --base_model, e.g. --base_model='smoked_salmons/awesome_llm'"
    

    # wandb configuration
    use_wandb = len(train_args.wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    
    if use_wandb:
        os.environ["WANDB_PROJECT"] = train_args.wandb_project
    if len(train_args.wandb_entity) > 0:
        os.environ["WANDB_ENTITY"] = train_args.wandb_entity
    if len(train_args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = train_args.wandb_watch
    if len(train_args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = train_args.wandb_log_model
    

    # load and preprocess data
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(train_args.base_model)
    tokenizer.pad_token_id = 0          # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"     # Allow batched inference
    train_data, val_data = load_and_preprocess_data(train_args, tokenizer)
    
    
    # load model and finetune
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = GPTNeoXForCausalLM.from_pretrained(
        train_args.base_model,
        quantization_config=bnb_config,
        device_map={"":0},
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=train_args.lora_r,
        lora_alpha=train_args.lora_alpha,
        target_modules=train_args.lora_target_modules,
        lora_dropout=train_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if train_args.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(train_args.resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(train_args.resume_from_checkpoint, "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
            train_args.resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=train_args.micro_batch_size,
        gradient_accumulation_steps=train_args.batch_size//train_args.micro_batch_size,
        num_train_epochs=train_args.num_epochs,
        learning_rate=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
        warmup_ratio=train_args.warmup_ratio,
        lr_scheduler_type=train_args.lr_scheduler_type,
        fp16=True,
        logging_steps=train_args.logging_steps,
        optim=train_args.optimizer,
        evaluation_strategy="steps" if train_args.val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=train_args.eval_steps if train_args.val_set_size > 0 else None,
        save_steps=train_args.eval_steps,
        output_dir=train_args.finetune_dir,
        save_total_limit=train_args.save_total_limit,
        load_best_model_at_end=True if train_args.val_set_size > 0 else False,
        ddp_find_unused_parameters=None,
        group_by_length=train_args.group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=train_args.wandb_run_name if use_wandb else None,
        )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=train_compute_metrics if train_args.use_compute_metrics else None,
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with torch.autocast("cuda"):
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    evaluation_result = trainer.evaluate(eval_dataset=val_data)
    print(evaluation_result)
 
    model.save_pretrained(train_args.finetune_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


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
    train(train_args)


if __name__ == "__main__":
    main()