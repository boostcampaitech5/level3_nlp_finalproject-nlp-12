from datasets import load_metric
from transformers import AutoTokenizer, GPTNeoXTokenizerFast, GPTNeoXForCausalLM
import numpy as np
from rouge import Rouge
import torch
from statistics import geometric_mean, mean


def compute_metrics(pred):
    base_model = 'nlpai-lab/kullm-polyglot-12.8b-v2'
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(base_model)
    
    # 사용할 metric을 불러옵니다.
    metric_perplexity = load_metric("perplexity")
    metric_bleu = load_metric("sacrebleu")
    metric_meteor = load_metric("meteor")
    metric_rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-3", "rouge-4", "rouge-5", "rouge-l"])
    metric_bertscore = load_metric("bertscore")

    # 학습에서 산출된 pred를 preds(모델이 생성)와 label(정답 데이터)로 분리합니다.
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    labels = np.where(pred.label_ids != -100, labels, tokenizer.pad_token_id)

    # 각 preds, labels 쌍 마다 score를 계산하고 저장하는 리스트 입니다. 
    ppl, bleu, meteor, rouge, bert = [], [], [], [], []

    for i in range(len(preds)) :
        # 숫자로 표현되어 있는 preds, labels 자연어로 decode 합니다. 
        # 이때, preds에는 프롬프트가 같이 생성이 됩니다. 따라서 "### 응답:" 이후로 생성되는 문장만 decode 합니다.
        decoded_preds = tokenizer.decode(preds[i], skip_special_tokens=True)
        decoded_labels = tokenizer.decode(labels[i], skip_special_tokens=True)
        if "### 응답:" in decoded_preds :
            decoded_preds = decoded_preds.split('### 응답:\n')[1][:-1]

        # score를 계산합니다. 각 compute 마다 주어져야 하는 predictions, references의 형식이 다름에 주의해 주십시오.
        ppl_score = metric_perplexity.compute(model_id='gpt2', add_start_token=False, input_texts=decoded_preds.split())['mean_perplexity']
        bleu_score = metric_bleu.compute(predictions=[decoded_preds], references=[[decoded_labels]])["score"]
        meteor_score = metric_meteor.compute(predictions=[decoded_preds], references=[decoded_labels])["meteor"]
        rouge_scores = metric_rouge.get_scores(decoded_preds, decoded_labels, avg=True)["rouge-l"]['f']
        bert_score = metric_bertscore.compute(predictions=[decoded_preds], references=[decoded_labels], lang='ko')["f1"][0]

        ppl.append(ppl_score)
        bleu.append(bleu_score/100)
        meteor.append(meteor_score)
        rouge.append(rouge_scores)
        bert.append(bert_score)

    # 각자 계산된 값을 산술평균 또는 기하평균 합니다.
    perplexity = mean(ppl)
    bleu_score = geometric_mean(bleu)
    meteor_score = geometric_mean(meteor)
    rouge_scores = geometric_mean(rouge)
    bert_score = geometric_mean(bert)


    return {
        'perplexity': round(perplexity, 5), 
        'sacre_bleu': round(bleu_score, 5),
        'meteor': round(meteor_score, 5),
        'rouge_l_f1': round(rouge_scores, 5),
        'bert_score_f1' : round(bert_score, 5),
    }
