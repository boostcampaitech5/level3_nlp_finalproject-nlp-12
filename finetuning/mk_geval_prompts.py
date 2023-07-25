import os
import sys
import time
import yaml
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.arguments import TrainArguments
from utils.prompter import Prompter


g_eval_prompt = """두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction)을 받게 될 것입니다. 지시문은 이전 대화내용을 포함하며 현재 대화에 대한 응답(Response)이 제시됩니다.
당신의 작업은 응답을 평가 단계에 따라 응답을 평가하는 것입니다. 
이 평가 기준을 꼼꼼히 읽고 이해하는 것이 중요합니다. 평가하는 동안 이 문서를 계속 열어두고 필요할 때 참조해 주세요. 

평가 기준:
- 이해 가능성 (0 - 1): Instruction에 기반하여 Response를 이해 할 수 있나요?
- 자연스러움 (1 - 5): 사람이 자연스럽게 말할 법한 Instruction 인가요?
- 맥락 유지 (1 - 5): Istruction을 고려했을 때 Response가 맥락을 유지하나요?
- 흥미롭기 (1 - 5): Response가 지루한가요, 아니면 흥미로운가요?
- Instruction 사용 (0 - 1): Instruction에 기반하여 Response를 생성 했나요?
- 공감 능력 (0 - 1): Response에 Instruction의 내용에 기반한 공감의 내용이 있나요?
- 대화 유도 (0 - 1): Response에 질문을 포함하여 사용자의 대답을 자연스럽게 유도하고 있나요?
- 전반적인 품질 (1 - 10): 위의 답변을 바탕으로 이 발언의 전반적인 품질에 대한 인상은 어떤가요?

평가 단계:
1. Instruction, 그리고 Response을 주의깊게 읽습니다.
2. 위의 평가 기준에 따라 Response을 엄격하게 평가합니다.

Instruction:
{{instruction}}

Response:
{{response}}


Result
- 이해 가능성 (0 - 1):
- 자연스러움 (1 - 5):
- 맥락 유지 (1 - 5):
- 흥미롭기 (1 - 5):
- Instruction 사용 (0 - 1):
- 공감 능력 (0 - 1):
- 대화 유도 (0 - 1): 
- 전반적인 품질 (1 - 10):"""


def inference(train_args: TrainArguments):
    start = time.time()

    MODEL = f"../finetuning/" + train_args.merge_dir
    prompter = Prompter(train_args.prompt_template_name)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model.eval()

    # load data
    if train_args.dataset_type == 'multi' :
        data = pd.read_csv('./test/test_dataset/test_multi.csv')
    elif train_args.dataset_type == 'single' :
        data = pd.read_csv('./test/test_dataset/test_single.csv')
    instruction = list(data['instruction'])


    def gen(x):
        inputs = tokenizer(
            x, 
            return_tensors='pt', 
            return_token_type_ids=False
        )
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()} # move to same device as model
        gened = model.generate(
            **inputs, 
            max_new_tokens=256,
            early_stopping=True,
            do_sample=True,
            eos_token_id=2,
        )

        gened_list = gened[0].tolist()
        
        try:
            eos_token_position = gened_list.index(2)
            gened_list = gened_list[:eos_token_position]
        except ValueError:
            pass
        
        output = tokenizer.decode(gened_list)
        print(output)
        response = output.split('### 응답:\n')[1]

        return response

    end = time.time()
    print(f'generation 하기 직전까지 걸린 시간 : {int(end-start)}초')


    g_eval_prompts = []
    for message in instruction:
        print('=============================================({0}/50)===================================================='.format(instruction.index(message)+1))
        final_string = ''
        prompt = prompter.generate_prompt(message)
        generations = gen(prompt)

        final_string = g_eval_prompt.replace("{{instruction}}", message)
        final_string = final_string.replace("{{response}}", generations)
        g_eval_prompts.append(final_string)

    df = pd.DataFrame({
        'g_eval_prompts' : g_eval_prompts,
    })
    df.to_csv('./test/result/'+train_args.save_file_name, index=False)


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
    inference(train_args)


if __name__ == "__main__":
    main()