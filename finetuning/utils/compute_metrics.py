from datasets import load_metric
from transformers import AutoTokenizer, GPTNeoXTokenizerFast
import numpy as np
from rouge import Rouge


def compute_metrics(pred):
    tokenizer = GPTNeoXTokenizerFast.from_pretrained('nlpai-lab/kullm-polyglot-12.8b-v2')

    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids

    if isinstance(preds, tuple):
        preds = preds[0]
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.decode(labels[0], skip_special_tokens=True)
    decoded_preds = tokenizer.decode(preds[0], skip_special_tokens=True)

    metric_bleu = load_metric("sacrebleu")
    metric_meteor = load_metric("meteor")
    metric_rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-3", "rouge-4", "rouge-5", "rouge-l"])
    metric_bertscore = load_metric("bertscore")

    bleu_score = metric_bleu.compute(predictions=[decoded_preds], references=[[decoded_labels]])["score"]
    meteor_score = metric_meteor.compute(predictions=[decoded_preds], references=[decoded_labels])["meteor"]
    rouge_scores = metric_rouge.get_scores(decoded_preds, decoded_labels, avg=True)["rouge-l"]['f']
    bert_score = metric_bertscore.compute(predictions=[decoded_preds], references=[decoded_labels], lang='ko')["f1"][0]

    return {
        'sacre_bleu': round(bleu_score/100, 5),
        'meteor': round(meteor_score, 5),
        'rouge_l_f1': round(rouge_scores, 5),
        'bert_score_f1' : round(bert_score, 5),
    }
