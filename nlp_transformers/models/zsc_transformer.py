import numpy as np

import torch
from transformers import AutoModelForSequenceClassification

from nlp_transformers import numpy_utils
from .base_transformer import BaseTransformer
from .training_mixin import TrainingMixin
from .model_outputs import ZeroShotClassificationOutput


__all__ = ['ZeroShotClassificationTransformer']


class ZeroShotClassificationTransformer(BaseTransformer, TrainingMixin):
    def __init__(self, pretrained_checkpoint):
        assert 'mnli' in pretrained_checkpoint or 'xnli' in pretrained_checkpoint
        super().__init__(pretrained_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_checkpoint)

        # set class ids
        self.entailment_id, self.contradiction_id = -1, -1
        for label, i in self.config.label2id.items():
            label = label.lower()
            if label.startswith('entail'):
                self.entailment_id = i
            elif label.startswith('contra'):
                self.contradiction_id = i

    def _create_sequence_pairs(self, sequences, labels, hypothesis_template):
        if isinstance(sequences, str):
            sequences = [sequences]

        # create sequence pairs where each pair contains sequence and label
        # the procedure creates new records where each label is paired with each sequence
        # number of sequence pairs = number of sequences * number of labels
        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)]
                                   for label in labels])

        return sequence_pairs

    def _postprocess_logits(self, logits: np.ndarray, no_sequences: int, no_labels: int, multi_label: bool):
        # reshape logits matrix
        logits = logits.reshape(no_sequences, no_labels, -1)

        if no_labels == 1:
            multi_label = True

        # compute probabilities of each label
        if not multi_label:
            # softmax the "entailment" logits over all candidate labels
            probs = numpy_utils.softmax(logits[..., self.entailment_id], axis=-1)
        else:
            # softmax over the entailment vs. contradiction dim for each label independently
            probs = numpy_utils.softmax(
                logits[..., [self.contradiction_id, self.entailment_id]], axis=-1)
            probs = probs[..., 1]
        return probs

    def predict_sample(self, sequences: [str, list], labels: list,
                       hypothesis_template='This example is {}.', *,
                       multi_label=True, max_inp_length=None, return_dict=False):
        """Run network inference for a small sample of data."""
        self.model.eval()

        # create sequence pairs (model input) by combining sequences with labels
        sequence_pairs = self._create_sequence_pairs(sequences, labels, hypothesis_template)

        # tokenize model input
        model_input = self.tokenizer(
            sequence_pairs, return_tensors='pt', max_length=max_inp_length,
            truncation='only_first', padding=True)
        model_input = model_input.to(self.model.device)

        # apply inference
        with torch.no_grad():
            logits = self.model(**model_input)[0].cpu().numpy()
        probs = self._postprocess_logits(logits, len(sequences), len(labels), multi_label)

        if return_dict:
            probs = [{labels[i]: x for i, x in enumerate(record)} for record in probs]

        return probs

    def tokenize_dataset(self, datasets, labels, hypothesis_template='This example is {}.', *,
                         inp_feature='inp', max_inp_length=None):
        """Tokenize dataset with input records before feeding them into the network."""
        def tokenize_records(records):
            inp = ['' if x is None else str(x) for x in records[inp_feature]]
            # note: create_sequence_pairs creates new records
            inp_pairs = self._create_sequence_pairs(inp, labels, hypothesis_template)
            model_inputs = self.tokenizer(
                inp_pairs, max_length=max_inp_length, truncation='only_first')

            # important: copy existing columns to the model_inputs
            for key in records.keys():
                model_inputs[key] = [records[key][i] for i in range(len(records[key]))
                                     for _ in range(len(labels))]

            return model_inputs

        return datasets.map(tokenize_records, batched=True)

    def create_dataset(self, x, *, labels, hypothesis_template='This example is {}.',
                       inp_feature='inp', max_inp_length=None, **kwargs):
        super().create_dataset(
            x, labels=labels, hypothesis_template=hypothesis_template,
            inp_feature=inp_feature, max_inp_length=max_inp_length)

    def predict(self, testset, labels, multi_label=True, *, output_dir='.', bs=64,
                log_level='passive', disable_tqdm=False, dataset_params={}):
        """Apply inference on test dataset, return predictions, labels (optionally) and probs."""
        testset = self.create_dataset(testset, labels=labels, **dataset_params)

        # create trainer with minimal setup for test-time
        trainer = self.get_trainer(
            output_dir=output_dir, bs=bs, log_level=log_level, disable_tqdm=disable_tqdm)

        # run inference
        out = trainer.predict(testset)

        # post-process outputs
        no_sequences = len(testset) // len(labels)
        assert len(labels) * no_sequences == len(testset)
        probs = self._postprocess_logits(out.predictions, no_sequences, len(labels), multi_label)

        return ZeroShotClassificationOutput(probs=probs, metrics=getattr(out, 'metrics', None))
