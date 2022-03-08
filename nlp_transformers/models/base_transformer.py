import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from datasets import Dataset


__all__ = ['BaseTransformer']


class BaseTransformer:
    def __init__(self, pretrained_checkpoint):
        self.pretrained_checkpoint = pretrained_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
        self.model = None

    def __str__(self):
        return f'{self.__class__.__name__}({self.pretrained_checkpoint})'

    def __repr__(self):
        return str(self)

    def from_pretrained(self, filename):
        self.model = self.model.from_pretrained(filename)
        self.tokenizer = self.tokenizer.from_pretrained(filename)
        return self

    def save_pretrained(self, filename):
        self.model.save_pretrained(filename)
        self.tokenizer.save_pretrained(filename)
        return self

    def num_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def config(self):
        return self.model.config if self.model is not None else None

    def get_token_length(self, x):
        model_input = self.tokenizer(x, max_length=None, truncation=False, padding=False)
        return np.array([len(x) for x in model_input.input_ids])

    def create_dataset(self, x, *args, inp_feature='inp', max_inp_length=None, **kwargs):
        # create dataset
        if isinstance(x, Dataset):
            dataset = x
        elif isinstance(x, dict):
            assert inp_feature in dict
            dataset = Dataset.from_dict(x)
        elif isinstance(x, list):
            dataset = Dataset.from_dict({inp_feature: x})
        elif isinstance(x, pd.Series):
            dataset = Dataset.from_dict({inp_feature: x.tolist()})
        elif isinstance(x, pd.DataFrame):
            dataset = Dataset.from_pandas(x)
        else:
            raise ValueError(f'Input parameter type "{type(x)}" is not supported.')

        # tokenize dataset if needed
        if 'input_ids' not in dataset.features:  # dataset is not tokenized
            params = dict(inp_feature=inp_feature, max_inp_length=max_inp_length, **kwargs)
            tokenized_dataset = self.tokenize_dataset(dataset, *args, **params)
        else:
            tokenized_dataset = dataset

        return tokenized_dataset

    def predict_sample(self, *args, **kwargs):
        raise NotImplementedError

    def tokenize_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
