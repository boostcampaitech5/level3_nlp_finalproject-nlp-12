from typing import List

from pydantic import BaseModel

class TrainArguments(BaseModel):
    # model/data params
    base_model: str = "beomi/polyglot-ko-12.8b-safetensors"
    data_path: str = "ohilikeit/empathetic_dialogues_kr"
    finetune_dir: str = ""
    merge_dir: str = ""
    prompt_template_name: str = "kullm-v2"
    # training hyperparams
    batch_size: int = 128
    micro_batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 512
    val_set_size: int = 200
    save_total_limit: int = 10
    logging_steps: int = 1
    optimizer: str = "adamw_torch"
    eval_steps: int = 10
    weight_decay: float = 0.
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = 'linear'
    resume_from_checkpoint: str = None
    use_compute_metrics: bool = False
    # lora hyperparams
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ["query_key_value", "xxx"]
    # llm hyperparams
    train_on_inputs: bool = True
    add_eos_token: bool = False
    group_by_length: bool = False
    # wandb params
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_watch: str = ""
    wandb_log_model: str = ""