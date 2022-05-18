from transformers import (
    BartForConditionalGeneration,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    MT5ForConditionalGeneration,
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
)

from .classification_transformer import ClassificationTransformer
from .feature_extraction_transformer import FeatureExtractionTransformer
from .translation_transformer import TranslationTransformer
from .zsc_transformer import ZeroShotClassificationTransformer

__all__ = [
    "ClassificationTransformer",
    "ZeroShotClassificationTransformer",
    "TranslationTransformer",
    "FeatureExtractionTransformer",
    "BERT",
    "DistilBERT",
    "RoBERTa",
    "T5",
    "MT5",
    "BART",
]

"""
Classification Transformers
"""


class BERT(ClassificationTransformer):
    PRETRAINED_CHECKPOINTS = [
        "bert-base-uncased",
        "bert-large-uncased",
        "bert-base-cased",
        "bert-large-cased",
        "bert-base-multilingual-uncased",
        "bert-base-multilingual-cased",
    ]

    def __init__(self, pretrained_checkpoint="bert-base-uncased", *args, **kwargs):
        super().__init__(pretrained_checkpoint, *args, **kwargs)
        assert isinstance(self.model, BertForSequenceClassification)


class DistilBERT(ClassificationTransformer):
    PRETRAINED_CHECKPOINTS = [
        "distilbert-base-uncased",
        "distilbert-base-cased",
        "distilgpt2",
        "distilbert-base-multilingual-cased",
    ]

    def __init__(
        self, pretrained_checkpoint="distilbert-base-uncased", *args, **kwargs
    ):
        super().__init__(pretrained_checkpoint, *args, **kwargs)
        assert isinstance(self.model, DistilBertForSequenceClassification)


class RoBERTa(ClassificationTransformer):
    PRETRAINED_CHECKPOINTS = ["roberta-base", "roberta-large", "distilroberta-base"]

    def __init__(self, pretrained_checkpoint="roberta-base", *args, **kwargs):
        super().__init__(pretrained_checkpoint, *args, **kwargs)
        assert isinstance(self.model, RobertaForSequenceClassification)


"""
Translation Transformers
"""


class T5(TranslationTransformer):
    PRETRAINED_CHECKPOINTS = ["t5-small", "t5-base", "t5-large", "t5-3B", "t5-11B"]

    def __init__(self, pretrained_checkpoint="t5-small"):
        super().__init__(pretrained_checkpoint)
        assert isinstance(self.model, T5ForConditionalGeneration)


class MT5(TranslationTransformer):
    PRETRAINED_CHECKPOINTS = [
        "google/mt5-small",
        "google/mt5-base",
        "google/mt5-large",
        "google/mt5-xl",
    ]

    def __init__(self, pretrained_checkpoint="google/mt5-small"):
        super().__init__(pretrained_checkpoint)
        assert isinstance(self.model, MT5ForConditionalGeneration)


class BART(TranslationTransformer):
    PRETRAINED_CHECKPOINTS = [
        "facebook/bart-base",
        "facebook/bart-large",
        "facebook/bart-large-mnli",
        "facebook/bart-large-cnn",
    ]

    def __init__(self, pretrained_checkpoint="facebook/bart-base"):
        super().__init__(pretrained_checkpoint)
        assert isinstance(self.model, BartForConditionalGeneration)
