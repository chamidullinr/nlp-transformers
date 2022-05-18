import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from transformers.trainer import logger
from transformers.trainer_pt_utils import torch_pad_and_concatenate

from .base_transformer import BaseTransformer
from .model_outputs import TranslationOutput
from .training_mixin import Seq2seqTrainingMixin

__all__ = ["TranslationTransformer"]


def torch_isin(element, test_elements):
    # note - torch.isin will be available in PyTorch=1.10
    cond = np.isin(element.cpu(), test_elements)
    cond = torch.Tensor(cond).to(torch.bool)
    return cond


class TranslationTransformer(BaseTransformer, Seq2seqTrainingMixin):
    SKIP_PROB_TOK = -1

    def __init__(self, pretrained_checkpoint):
        super().__init__(pretrained_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_checkpoint)

    def _generate(self, input_ids, output_probs, **kwargs):
        self.model.eval()
        if output_probs:
            with torch.no_grad():
                out_dict = self.model.generate(
                    input_ids,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **kwargs,
                )
            logits = out_dict["sequences"].cpu()
            scores = torch.cat([x.cpu().unsqueeze(1) for x in out_dict["scores"]], 1)
            probs, _ = F.softmax(scores, -1).max(-1)
            # set prob of special tokens to -1
            probs[
                torch_isin(logits[:, 1:], self.tokenizer.all_special_ids)
            ] = self.SKIP_PROB_TOK
        else:
            with torch.no_grad():
                logits = self.model.generate(input_ids, **kwargs).cpu()
            probs = None
        return logits, probs

    def predict_sample(
        self,
        x,
        *,
        max_inp_length=None,
        max_trg_length=None,
        output_probs=False,
        **kwargs,
    ):
        """Run network inference for a small sample of data."""
        model_input = self.tokenizer(
            x,
            return_tensors="pt",
            max_length=max_inp_length,
            truncation=True,
            padding=True,
        )
        model_input = model_input.to(self.model.device)
        logits, probs = self._generate(
            **model_input,
            output_probs=output_probs,
            max_length=max_trg_length,
            **kwargs,
        )
        decoded = self.tokenizer.batch_decode(logits, skip_special_tokens=True)
        if len(decoded) == 1:
            decoded = decoded[0]
        out = decoded if probs is None else (decoded, probs)
        return out

    def tokenize_dataset(
        self,
        datasets,
        *,
        inp_feature="inp",
        trg_feature="trg",
        max_inp_length=None,
        max_trg_length=None,
        prefix=None,
        prefix_col=None,
    ):
        """Tokenize dataset with input and target records
        before feeding them into the network.
        """

        def tokenize_records(records):
            inp = ["" if x is None else str(x) for x in records[inp_feature]]
            if prefix is not None:
                inp = [f"{prefix} {x}" for x in inp]
            elif prefix_col is not None and prefix_col in records:
                inp = [f"{pref} {x}" for x, pref in zip(inp, records[prefix_col])]
            model_inputs = self.tokenizer(
                inp, max_length=max_inp_length, truncation=True
            )
            if trg_feature is not None and trg_feature in records:
                trg = ["" if x is None else str(x) for x in records[trg_feature]]
                with self.tokenizer.as_target_tokenizer():
                    model_inputs["labels"] = self.tokenizer(
                        trg, max_length=max_trg_length, truncation=True
                    ).input_ids
            else:
                # due to a shortcoming in transformers==4.8.2
                # where 'labels' cannot be empty
                model_inputs["labels"] = model_inputs.input_ids
            return model_inputs

        tokenized_datasets = datasets.map(tokenize_records, batched=True)
        return tokenized_datasets

    def _predict(self, testset, trainer, max_length, output_probs):
        # create PyTorch dataloader
        dataloader = trainer.get_test_dataloader(testset)

        start_time = time.time()

        # get model
        model = trainer._wrap_model(trainer.model, training=False)
        max_length = max_length or model.config.max_length

        # set mixed precision - fp16 (make sure it isn't called while training)
        if not trainer.is_in_train and trainer.args.fp16_full_eval:
            model = model.half().to(trainer.args.device)

        # log prediction parameters
        logger.info("***** Running Prediction *****")
        logger.info(f"  Num examples = {len(testset)}")
        logger.info(f"  Batch size = {dataloader.batch_size}")

        # main evaluation loop
        model.eval()
        trainer.callback_handler.eval_dataloader = dataloader
        logits_all, probs_all, labels_all = None, None, None
        for step, inputs in enumerate(dataloader):
            inputs = trainer._prepare_inputs(inputs)

            # apply inference
            logits, probs = self._generate(
                inputs["input_ids"],
                output_probs,
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
            )

            # get labels
            labels = inputs.get("labels")
            if labels is not None and labels.shape[-1] < max_length:
                labels = trainer._pad_tensors_to_max_len(labels, max_length)

            # store variables
            logits = logits.cpu()
            logits_all = (
                logits
                if logits_all is None
                else torch_pad_and_concatenate(
                    logits_all, logits, padding_index=self.tokenizer.pad_token_id
                )
            )
            if probs is not None:
                probs = probs.cpu()
                probs_all = (
                    probs
                    if probs_all is None
                    else torch_pad_and_concatenate(
                        probs_all, probs, padding_index=self.SKIP_PROB_TOK
                    )
                )
            if labels is not None:
                labels = labels.cpu()
                labels_all = (
                    labels
                    if labels_all is None
                    else torch_pad_and_concatenate(
                        labels_all, labels, padding_index=self.tokenizer.pad_token_id
                    )
                )

            trainer.control = trainer.callback_handler.on_prediction_step(
                trainer.args, trainer.state, trainer.control
            )

        # convert to numpy
        logits_all = logits_all.numpy()
        if probs_all is not None:
            probs_all = probs_all.numpy()
        if labels_all is not None:
            labels_all = labels_all.numpy()

        # compute metrics
        runtime = time.time() - start_time
        num_samples = len(testset)
        num_steps = math.ceil(num_samples / dataloader.batch_size)
        samples_per_second = num_samples / runtime
        steps_per_second = num_steps / runtime
        metrics = {
            "test_runtime": round(runtime, 4),
            "test_samples_per_second": round(samples_per_second, 3),
            "test_steps_per_second": round(steps_per_second, 3),
        }

        return logits_all, probs_all, labels_all, metrics

    def predict(
        self,
        testset,
        *,
        output_dir=".",
        bs=64,
        log_level="passive",
        disable_tqdm=False,
        output_probs=False,
        output_text_preds=False,
        max_length=None,
        dataset_params={},
    ):
        """
        Apply inference on test dataset,
        return predictions, labels (optionally) and probs (optionally).
        """
        testset = self.create_dataset(testset, **dataset_params)

        # create trainer with minimal setup for test-time
        trainer = self.get_trainer(
            output_dir=output_dir, bs=bs, log_level=log_level, disable_tqdm=disable_tqdm
        )

        # run inference
        logits, probs, labels, metrics = self._predict(
            testset, trainer, max_length, output_probs
        )

        # include optional output values
        opt_kwargs = {}
        if output_text_preds:
            opt_kwargs["text_predictions"] = self.tokenizer.batch_decode(
                logits, skip_special_tokens=True
            )

        return TranslationOutput(
            predictions=logits,
            label_ids=labels,
            probs=probs,
            metrics=metrics,
            **opt_kwargs,
        )
