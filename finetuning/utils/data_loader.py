from datasets import load_dataset
from transformers import GPTNeoXTokenizerFast
from utils.prompter import Prompter
from utils.arguments import TrainArguments


def load_and_preprocess_data(train_args: TrainArguments, tokenizer: GPTNeoXTokenizerFast):

    if train_args.data_path.endswith(".json") or train_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=train_args.data_path)
    else:
        data = load_dataset(train_args.data_path)

    prompter = Prompter(template_name = train_args.prompt_template_name,
                        verbose = False)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=train_args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < train_args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["유저"],
            data_point["챗봇"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        user_prompt = prompter.generate_prompt(data_point["유저"])
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=train_args.add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"]) - int(train_args.add_eos_token)
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        
        return tokenized_full_prompt

    if train_args.val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=train_args.val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    return train_data, val_data