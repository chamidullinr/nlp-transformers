from typing import NamedTuple, Optional

import numpy as np


__all__ = ['ClassificationOutput', 'ZeroShotClassificationOutput', 'TranslationOutput',
           'FeatureExtractionOutput']


class ClassificationOutput(NamedTuple):
    """Output object returned by `transformer.predict` method."""
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    probs: Optional[np.ndarray]
    metrics: Optional[dict]
    logits: Optional[np.ndarray] = None


class ZeroShotClassificationOutput(NamedTuple):
    """Output object returned by `transformer.predict` method."""
    probs: np.ndarray
    metrics: Optional[dict]


class TranslationOutput(NamedTuple):
    """Output object returned by `transformer.predict` method."""
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    probs: Optional[np.ndarray]
    metrics: Optional[dict]
    text_predictions: Optional[np.ndarray] = None


class FeatureExtractionOutput(NamedTuple):
    """Output object returned by `transformer.predict` method."""
    embeddings: np.ndarray
    metrics: Optional[dict]
