from datasets import load_metric
from transformers import GPTNeoXTokenizerFast, GPTNeoXForCausalLM
import numpy as np
from rouge import Rouge
from statistics import geometric_mean
import torch
from tqdm import tqdm


def train_compute_metrics(pred):

    model = GPTNeoXForCausalLM.from_pretrained('nlpai-lab/kullm-polyglot-12.8b-v2')

    logits = torch.tensor(pred.predictions.argmax(-1).flatten(), dtype=torch.int64)
    logits = logits.unsqueeze(0)  # torch.Size([1, 35200])

    max_length = model.config.max_position_embeddings # 2048
    stride = 1024
    seq_len = logits.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # 마지막 루프의 스트라이드 값과 다를 수 있음
        input_ids = logits[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # 손실은 모든 유효한 레이블에 대한 평균값을 구하는 교차 엔트로피(cross entropy)로 계산됩니다.
            # 나이브 베이지안 모델은 내부적으로 레이블을 왼쪽으로 1개씩 밀기 때문에, (타켓 - 1)개 만큼의 레이블에 대해 손실을 계산합니다.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    return {'perplexity':ppl}


def test_compute_metrics(pred):
    tokenizer = GPTNeoXTokenizerFast.from_pretrained('nlpai-lab/kullm-polyglot-12.8b-v2')

    # 사용할 metric을 불러옵니다.
    metric_bleu = load_metric("sacrebleu")
    metric_meteor = load_metric("meteor")
    metric_rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-3", "rouge-4", "rouge-5", "rouge-l"])
    metric_bertscore = load_metric("bertscore")

    # 학습에서 산출된 pred를 preds(모델이 생성)와 label(정답 데이터)로 분리합니다.
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    labels = np.where(pred.label_ids != -100, labels, tokenizer.pad_token_id)

    scores = {
        'sacre_bleu': [],
        'meteor': [],
        'rouge_l_f1': [],
        'bert_score_f1': [],
    }

    for i in range(len(preds)):
        decoded_preds = tokenizer.decode(preds[i], skip_special_tokens=True)
        decoded_labels = tokenizer.decode(labels[i], skip_special_tokens=True)
        if "### 응답:" in decoded_preds:
            decoded_preds = decoded_preds.split('### 응답:\n')[1][:-1]

        bleu_score = metric_bleu.compute(predictions=[decoded_preds], references=[[decoded_labels]])["score"]
        meteor_score = metric_meteor.compute(predictions=[decoded_preds], references=[decoded_labels])["meteor"]
        rouge_scores = metric_rouge.get_scores(decoded_preds, decoded_labels, avg=True)["rouge-l"]['f']
        bert_score = metric_bertscore.compute(predictions=[decoded_preds], references=[decoded_labels], lang='ko')["f1"][0]

        scores['sacre_bleu'].append(bleu_score / 100)
        scores['meteor'].append(meteor_score)
        scores['rouge_l_f1'].append(rouge_scores)
        scores['bert_score_f1'].append(bert_score)

    scores = {k: geometric_mean(v) for k, v in scores.items()}

    return {k: round(v, 5) for k, v in scores.items()}