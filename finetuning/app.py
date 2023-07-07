import torch
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

def load_model():
    """Load the local Hugging Face model."""
    MODEL = "lora_merged_model" 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

pipe = pipeline(
    'text-generation',
    model = model,
    tokenizer = tokenizer,
    device=0
)

def answer(state, state_chatbot, text):
    messages = state + [{'role' : '명령어', 'content' : text}]

    conversation_history = '\n'.join(
        [f"### {msg['role']}:\n{msg['content']}" for msg in messages]
    )
    print(conversation_history)

    ans = pipe(
        conversation_history + '\n### 응답:',
        do_sample = True,
        max_new_tokens = 512,
        temperature = 0.7,
        top_p = 0.9,
        return_full_text = False,
        eos_token_id = 2,
    )

    msg = ans[0]['generated_text']
    if '###' in msg:
        msg = msg.split('###')[0]
    
    new_state = [{'role' : '이전 명령어', 'content' : text},
                 {'role' : '이전 응답', 'content' : msg}]
    
    state = state + new_state
    state_chatbot = state_chatbot + [[text, msg]]

    print('state : ', state)
    print('state_chatbot : ', state_chatbot)

    return state, state_chatbot, state_chatbot


with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State(
        [
            {
                'role' : '맥락',
                'content' : '공감을 잘하는 챗봇 "내마리" 입니다. 사람들의 질문에 친절하게 답변해줍니다.'
            },
            {
                'role' : '맥락',
                'content' : '제 목표는 유용하고 친절하며 재미있는 챗봇이 되는 것입니다. 저에게 조언을 구하거나 답변을 요청하거나 무슨 일이든 이야기할 수 있습니다.'
            },
            {
                "role": "명령",
                "content": "아래는 일상적 대화에서의 질문입니다. 최대 3문장의 간결하고 친절한 응답을 만들어주세요",
            },
        ]
    )
    state_chatbot = gr.State([])
    
    with gr.Row():
        gr.Markdown("<h3><center>공감능력 개쩌는 부덕이와의 대화</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        txt = gr.Textbox(
            label="어떤 것을 묻고 싶나요?",
            placeholder="찐 F인 부덕이에게 어떤 말이든 자유롭게 해주세요!",
            lines=1,
        ).style(container = False)

    gr.HTML("Demo application of a Polyglot-ko, Huggingface")
    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot])
    txt.submit(lambda: "", None, txt)

demo.launch(debug=True, share=True)
