import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from nlp_transformers import numpy_utils

from .base_transformer import BaseTransformer
from .model_outputs import ClassificationOutput
from .training_mixin import TrainingMixin

__all__ = ["ClassificationTransformer"]


class ClassificationTransformer(BaseTransformer, TrainingMixin):
    def __init__(self, pretrained_checkpoint, classes=None):
        super().__init__(pretrained_checkpoint)

        # create params dictionary
        params = dict()
        if classes is not None:
            params["num_labels"] = len(classes)
            params["id2label"] = {i: x for i, x in enumerate(classes)}
            params["label2id"] = {x: i for i, x in enumerate(classes)}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_checkpoint, **params
        )

    def id2label(self, ids):
        return np.array([self.config.id2label[x] for x in ids])

    def label2id(self, labels):
        return np.array([self.config.label2id[x] for x in labels])

    def predict_sample(self, x, *, max_inp_length=None, return_dict=False):
        """Run network inference for a small sample of data."""
        self.model.eval()
        model_input = self.tokenizer(
            x,
            return_tensors="pt",
            max_length=max_inp_length,
            truncation=True,
            padding=True,
        )
        model_input = model_input.to(self.model.device)

        with torch.no_grad():
            logits = self.model(**model_input).logits.cpu()
        probs = F.softmax(logits, dim=-1).numpy()

        if return_dict:
            probs = [
                {self.config.id2label[i]: x for i, x in enumerate(record)}
                for record in probs
            ]

        return probs

    def tokenize_dataset(
        self, datasets, *, inp_feature="inp", trg_feature="trg", max_inp_length=None
    ):
        """Tokenize dataset with input and target records
        before feeding them into the network.
        """

        def tokenize_records(records):
            inp = ["" if x is None else str(x) for x in records[inp_feature]]
            model_inputs = self.tokenizer(
                inp, max_length=max_inp_length, truncation=True
            )
            if trg_feature is not None and trg_feature in records:
                model_inputs["labels"] = [
                    self.config.label2id[x] for x in records[trg_feature]
                ]
            return model_inputs

        return datasets.map(tokenize_records, batched=True)

    def predict(
        self,
        testset,
        *,
        output_dir=".",
        bs=64,
        output_logits=False,
        temperature=1,
        log_level="passive",
        disable_tqdm=False,
        dataset_params={}
    ):
        """Apply inference on test dataset
        and return predictions, labels (optionally) and probs.
        """
        testset = self.create_dataset(testset, **dataset_params)

        # create trainer with minimal setup for test-time
        trainer = self.get_trainer(
            output_dir=output_dir, bs=bs, log_level=log_level, disable_tqdm=disable_tqdm
        )

        # run inference
        out = trainer.predict(testset)
        logits, targs = out.predictions, out.label_ids
        probs = numpy_utils.softmax(out.predictions, axis=-1, temperature=temperature)
        preds = probs.argmax(-1)

        # include optional output values
        opt_kwargs = {}
        if output_logits:
            opt_kwargs["logits"] = logits

        return ClassificationOutput(
            predictions=preds,
            label_ids=targs,
            probs=probs,
            metrics=getattr(out, "metrics", None),
            **opt_kwargs
        )
