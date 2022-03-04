import math
import time

import numpy as np

import torch
from transformers import AutoModel
from transformers.trainer import logger
from transformers.trainer_pt_utils import torch_pad_and_concatenate

from nlp_transformers import numpy_utils
from .base_transformer import BaseTransformer
from .training_mixin import TrainingMixin
from .model_outputs import FeatureExtractionOutput


__all__ = ['FeatureExtractionTransformer']


# def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     embeddings = (torch.sum(token_embeddings * input_mask_expanded, 1) /
#                  torch.clamp(input_mask_expanded.sum(1), min=1e-9))
#     embeddings = F.normalize(embeddings, p=2, dim=1)
#     return embeddings


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray):
    # thanks to https://www.sbert.net/
    attention_mask_expanded = np.broadcast_to(
        attention_mask.reshape(*attention_mask.shape, -1), token_embeddings.shape)
    sequence_length = np.clip(attention_mask_expanded.sum(1), a_min=1e-9, a_max=np.inf)
    embeddings = (token_embeddings * attention_mask_expanded).sum(1) / sequence_length
    return embeddings


class FeatureExtractionTransformer(BaseTransformer, TrainingMixin):
    def __init__(self, pretrained_checkpoint):
        super().__init__(pretrained_checkpoint)
        self.model = AutoModel.from_pretrained(pretrained_checkpoint)

    def predict_sample(self, x, *, max_inp_length=None, output_logits=False,
                       normalize_embeddings=False):
        """
        Run network inference for a small sample of data.

        Note: If normalize_embeddings is True, then embeddings can be compared with dot product.
        """
        self.model.eval()
        model_input = self.tokenizer(
            x, return_tensors='pt', max_length=max_inp_length,
            truncation=True, padding=True)
        model_input = model_input.to(self.model.device)

        with torch.no_grad():
            logits = self.model(**model_input).last_hidden_state.cpu().numpy()

        # create and normalize embeddings
        embeddings = mean_pooling(logits, model_input['attention_mask'].cpu().numpy())
        if normalize_embeddings:
            embeddings = numpy_utils.normalize(embeddings, p=2, axis=1)

        return (embeddings, logits) if output_logits else embeddings

    def tokenize_dataset(self, datasets, *, inp_feature='inp', max_inp_length=None):
        """Tokenize dataset with input records before feeding them into the network."""
        def tokenize_records(records):
            inp = ['' if x is None else str(x) for x in records[inp_feature]]
            model_inputs = self.tokenizer(inp, max_length=max_inp_length, truncation=True)
            return model_inputs

        return datasets.map(tokenize_records, batched=True)

    def _predict(self, testset, trainer, normalize_embeddings):
        # create PyTorch dataloader
        dataloader = trainer.get_test_dataloader(testset)

        start_time = time.time()

        # get model
        model = trainer._wrap_model(self.model, training=False)

        # set mixed precision - fp16 (make sure it isn't called while training)
        if not trainer.is_in_train and trainer.args.fp16_full_eval:
            model = model.half().to(trainer.args.device)

        # log prediction parameters
        logger.info(f'***** Running Prediction *****')
        logger.info(f'  Num examples = {len(testset)}')
        logger.info(f'  Batch size = {dataloader.batch_size}')

        # main evaluation loop
        model.eval()
        trainer.callback_handler.eval_dataloader = dataloader
        embeddings_all = []
        for step, inputs in enumerate(dataloader):
            inputs = trainer._prepare_inputs(inputs)

            # apply inference
            with torch.no_grad():
                logits = model(**inputs).last_hidden_state.cpu().numpy()
                if isinstance(logits, tuple):
                    logits = logits[0]

            # create sequence embeddings (embedding vectors for each input sequence)
            # note: logits are token embeddings (embedding vectors for each token in input sequence)
            no_records, input_length, emb_size = logits.shape
            embeddings = mean_pooling(logits, inputs['attention_mask'].cpu().numpy())
            if normalize_embeddings:
                embeddings = numpy_utils.normalize(embeddings, p=2, axis=1)
            embeddings_all.append(embeddings)

            trainer.control = trainer.callback_handler.on_prediction_step(
                trainer.args, trainer.state, trainer.control)

        # convert to numpy
        embeddings_all = np.concatenate(embeddings_all, axis=0)

        # compute metrics
        runtime = time.time() - start_time
        num_samples = len(testset)
        num_steps = math.ceil(num_samples / dataloader.batch_size)
        samples_per_second = num_samples / runtime
        steps_per_second = num_steps / runtime
        metrics = {
            'test_runtime': round(runtime, 4),
            'test_samples_per_second': round(samples_per_second, 3),
            'test_steps_per_second': round(steps_per_second, 3)}

        return embeddings_all, metrics

    def predict(self, testset, *, output_dir='.', bs=64,
                log_level='passive', disable_tqdm=False,
                normalize_embeddings=False, dataset_params={}):
        """
        Apply inference on test dataset, return predictions, labels (optionally) and probs.

        Note: If normalize_embeddings is True, then embeddings can be compared with dot product.
        """
        testset = self.create_dataset(testset, **dataset_params)

        # create trainer with minimal setup for test-time
        trainer = self.get_trainer(
            output_dir=output_dir, bs=bs, log_level=log_level, disable_tqdm=disable_tqdm)

        # run inference
        embeddings, metrics = self._predict(testset, trainer, normalize_embeddings)

        return FeatureExtractionOutput(embeddings=embeddings, metrics=metrics)
