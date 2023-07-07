import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompter import Prompter

start = time.time()

MODEL = "./lora_merged_model/test"
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model.eval()

prompter = Prompter("kullm_test")

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

messages = [
            '오늘 날씨가 맑아서 기분이 좋아요 !', 
            '내일 시험인데 너무 떨려서 잠이 안와요. 어떡하죠?',
            '요즘 스트레스 때문에 자주 불면증에 시달리고 피곤해요.',
            '나는 커서 연기를 하는 배우가 될거야! 그게 나의 꿈이야',
            '이제 일 그만하고 집에가서 쉬고싶어, 넘 힘들어 ㅠㅠ'
            ]

for message in messages:
    start = time.time()
    prompt = prompter.generate_prompt(message)
    gen(prompt)
    end = time.time()
    print(f'걸린 시간 : {int(end-start)}초')
    print('##################################################################')
    print('##################################################################')