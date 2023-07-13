import sys
import time
import yaml

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.arguments import TrainArguments
from utils.prompter import Prompter


def inference(train_args: TrainArguments):
    start = time.time()

    MODEL = train_args.merge_dir
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model.eval()

    prompter = Prompter(train_args.prompt_template_name)

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
        
        print(tokenizer.decode(gened_list))

    end = time.time()
    print(f'generation 하기 직전까지 걸린 시간 : {int(end-start)}초')

    messages = ['오늘 날씨가 맑아서 기분이 좋아요!',
                '내일 시험인데 너무 떨려서 잠이 안와요. 어떡하죠?',
                '요즘 스트레스 때문에 자주 불면증에 시달리고 피곤해요.',
                '나는 커서 연기를 하는 배우가 될거야! 그게 나의 꿈이야',
                '이제 일 그만하고 집에가서 쉬고싶어, 넘 힘들어 ㅠㅠ',
                '가족과 오랜만에 여행을 가려고 하는데 좋은 장소 추천해주세요.',
                '새로 시작한 직장에서 복잡한 인간 관계 때문에 힘들어요.',
                '사실 제가 정확히 무슨 목표를 가지고 있는지 잘 모르겠어요.',
                '공부하다가 문득, 왜 이렇게 열심히 해야 하는지 의문이 들어요.',
                '요즘에 계속 머리가 아파서 고민이에요.',
                '날이 좋아서 친구들과 함께 소풍을 갔어요. 햇살이 너무 좋았어요!', 
                '새로 산 피아노로 연습하는데, 손가락이 잘 안 따라가요.',
                '주말에 본 영화가 너무 재밌어서 아직도 생각나네요.',
                '요즘 가족들과 사이가 좋지 않아서 스트레스 받아요.',
                '저는 어릴 때부터 음악에 관심이 많아서 지금은 밴드 활동을 하고 있어요.'
                ]

    for message in messages:
        start = time.time()
        prompt = prompter.generate_prompt(message)
        gen(prompt)
        end = time.time()
        print(f'걸린 시간 : {int(end-start)}초')
        print('##################################################################')

    
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