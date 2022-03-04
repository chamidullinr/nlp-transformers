from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def accuracy(pred: np.array, targ: np.array):
    if len(pred.shape) == 2:
        pred = pred.argmax(1)
    return accuracy_score(targ, pred)


def f1(pred: np.array, targ: np.array, labels=None):
    if len(pred.shape) == 2:
        if labels is None:
            labels = np.arange(pred.shape[1])
        pred = pred.argmax(1)
    return f1_score(targ, pred, labels=labels, average='macro')


def text_accuracy(a: [str, Iterable], b: [str, Iterable]):
    def metric(_a, _b):
        return int(_a == _b)
    
    if isinstance(a, str) and isinstance(b, str):
        out = metric(a, b)
    else:
        out = np.mean([metric(_a, _b) for _a, _b in zip(a, b)])
    return out


def levenshtein_score(a: [str, Iterable], b: [str, Iterable]):
    import stringdist

    def metric(_a, _b):
        return 1 - stringdist.levenshtein_norm(_a, _b)

    if isinstance(a, str) and isinstance(b, str):
        out = metric(a, b)
    else:
        out = np.mean([metric(_a, _b) for _a, _b in zip(a, b)])
    return out


def jaccard_index(a: [str, Iterable], b: [str, Iterable]):
    def metric(_a, _b):
        label_1 = set(_a.split(' ')) if not isinstance(_a, list) else set(_a)
        label_2 = set(_b.split(' ')) if not isinstance(_b, list) else set(_b)
        union_len = len(label_1.union(label_2))
        intersection_len = len(label_1.intersection(label_2))
        return intersection_len / union_len

    if isinstance(a, str) and isinstance(b, str):
        out = metric(a, b)
    else:
        out = np.mean([metric(_a, _b) for _a, _b in zip(a, b)])
    return out


class ClassificationMetricsCallback:
    def __init__(self, metrics=[accuracy, f1]):
        self.metrics = metrics

    def __call__(self, eval_preds):
        preds, labels = eval_preds

        # compute metrics
        out = {met.__name__: met(preds, labels) for met in self.metrics}
        
        return out


class TranslationMetricsCallback:
    def __init__(self, tokenizer, metrics=[text_accuracy, levenshtein_score, jaccard_index]):
        self.tokenizer = tokenizer
        self.metrics = metrics

    def __call__(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # replace -100 with pad token in the labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # decode preds and labels
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # post-process decoded strings
        decoded_preds = [x.strip() for x in decoded_preds]
        decoded_labels = [x.strip() for x in decoded_labels]

        # compute metrics
        out = {met.__name__: met(decoded_preds, decoded_labels) for met in self.metrics}

        return out
