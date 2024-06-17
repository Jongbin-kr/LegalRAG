import string
import re
from collections import Counter
import re
from typing import *

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))


def f1(decoded_preds, decoded_labels):
    f1_all = []
    for prediction, answers in zip(decoded_preds, decoded_labels):
        if type(answers) == list:
            if len(answers) == 0:
                return 0
            f1_all.append(np.max([qa_f1_score(prediction, gt)
                          for gt in answers]))
        else:
            f1_all.append(qa_f1_score(prediction, answers))
    
    f1_score = 100 * np.mean(f1_all)
    print(f"f1 score: {f1_score}")
    return f1_score


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



def calculate_bleu_scores(preds, groundtruths):
    smoothing = SmoothingFunction().method4

    bleu_scores = [sentence_bleu([gt.split()], pred.split()) for pred, gt in zip(preds, groundtruths)]
    bleu1_scores = [sentence_bleu([gt.split()], pred.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing) for pred, gt in zip(preds, groundtruths)]
    bleu2_scores = [sentence_bleu([gt.split()], pred.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing) for pred, gt in zip(preds, groundtruths)]

    bleu_score = 100 * np.mean(bleu_scores)
    bleu1_score = 100 * np.mean(bleu1_scores)
    bleu2_score = 100 * np.mean(bleu2_scores)

    print(f"BLEU Score: {bleu_score}%")
    print(f"BLEU-1 Score: {bleu1_score}%")
    print(f"BLEU-2 Score: {bleu2_score}%")
    
    return bleu_score, bleu1_score, bleu2_score

def calculate_rouge_scores(preds, groundtruths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, gt in zip(preds, groundtruths):
        scores = scorer.score(gt, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    rouge1_score = 100 * np.mean(rouge1_scores)
    rouge2_score = 100 * np.mean(rouge2_scores)
    rougeL_score = 100 * np.mean(rougeL_scores)

    print(f"ROUGE-1 Score: {rouge1_score}%")
    print(f"ROUGE-2 Score: {rouge2_score}%")
    print(f"ROUGE-L Score: {rougeL_score}%")
    
    return rouge1_score, rouge2_score, rougeL_score


def calcuate_laws_match(preds, groundtruths):
    true_positive, false_positive, false_negative = 0, 0, 0
    for pred, groundtruth in zip(preds, groundtruths):
        if isinstance(pred, float):
            continue
        
        pred = set(eval(pred))
        groundtruth = set(groundtruth.split(', '))

        true_positive += len(pred &  groundtruth)
        false_positive += len(pred - groundtruth)
        false_negative += len(groundtruth - pred)
        
    print(f"true_positive: {true_positive}, false_positive: {false_positive}, false_negative: {false_negative}")
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"precision: {precision}, recall: {recall}, f1: {f1}")
    
    return precision, recall, f1

## some helper functions
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def find_entity_tags(sentence):
    entity_regex = r'(.+?)(?=\s<|$)'
    tag_regex = r'<(.+?)>'
    entity_names = re.findall(entity_regex, sentence)
    tags = re.findall(tag_regex, sentence)

    results = {}
    for entity, tag in zip(entity_names, tags):
        if "<" in entity:
            results[entity.split("> ")[1]] = tag
        else:
            results[entity] = tag
    return results

def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0

def extract_law_names(text) -> List[str]:
    # 입력값이 NaN이나 None인 경우 None 반환.
    if pd.isna(text):
        return np.NaN
    # 정규 표현식으로 법령명 추출
    law_names = re.findall(r'\b(?:[\w가-힣]+법|법률|규칙|조례|명령|규정)\s제[\d가-힣]+조\b', str(text))
    return law_names