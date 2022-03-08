import math
import time

import numpy as np
import torch

from transformers.trainer import logger

from nlp_transformers import numpy_utils
from .classification_transformer import ClassificationTransformer
from .special.modeling_distilbert import DistilBertForMultiheadSequenceClassification
from .model_outputs import ClassificationOutput

__all__ = ['MultiheadSequenceClassification']


class MultiheadSequenceClassification(ClassificationTransformer):
    def __init__(self, pretrained_checkpoint, classes_dict: dict):
        super().__init__(pretrained_checkpoint)
        self.classes_dict = classes_dict
        self.cls_heads = {name: len(classes) for name, classes in classes_dict.items()}

        # create params dictionary
        params = dict()
        # params['num_labels'] = self.cls_heads
        params['id2label'], params['label2id'] = {}, {}
        for name, classes in classes_dict.items():
            params['id2label'][name] = {i: x for i, x in enumerate(classes)}
            params['label2id'][name] = {x: i for i, x in enumerate(classes)}

        self.model = DistilBertForMultiheadSequenceClassification.from_pretrained(
            pretrained_checkpoint, cls_heads=self.cls_heads, **params)

    def id2label(self, ids, cls_name=None):
        assert cls_name in self.cls_heads
        return np.array([self.config.id2label[cls_name][x] for x in ids])

    def label2id(self, labels, cls_name=None):
        assert cls_name in self.cls_heads
        return np.array([self.config.label2id[cls_name][x] for x in labels])

    def _predict(self, testset, trainer):
        # create PyTorch dataloader
        dataloader = trainer.get_test_dataloader(testset)

        start_time = time.time()

        # get model
        model = trainer._wrap_model(trainer.model, training=False)

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
        logits_all, labels_all = [], []
        for step, inputs in enumerate(dataloader):
            inputs = trainer._prepare_inputs(inputs)

            # apply inference
            with torch.no_grad():
                logits = self.model(**inputs).logits  # .cpu().numpy()
            if isinstance(logits, dict):
                logits = {k: v.cpu().numpy() for k, v in logits.items()}
            else:
                logits = logits.cpu().numpy()

            # store results
            logits_all.append(logits)
            if 'labels' in inputs:
                labels_all.append(inputs['labels'].cpu().numpy())

            trainer.control = trainer.callback_handler.on_prediction_step(
                trainer.args, trainer.state, trainer.control)

        # convert to numpy
        if isinstance(logits_all[0], dict):
            _logits_all = {}
            for k in logits_all[0].keys():
                _logits_all[k] = np.concatenate([x[k] for x in logits_all], axis=0)
            logits_all = _logits_all
        else:
            logits_all = np.concatenate(logits_all, axis=0)
        if len(labels_all) > 0:
            labels_all = np.concatenate(labels_all, axis=0)

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

        return logits_all, labels_all, metrics

    def tokenize_dataset(self, datasets, *, inp_feature='inp', trg_feature='trg',
                         max_inp_length=None, cls_name=None):
        """Tokenize dataset with input and target records before feeding them into the network."""
        def tokenize_records(records):
            inp = ['' if x is None else str(x) for x in records[inp_feature]]
            model_inputs = self.tokenizer(inp, max_length=max_inp_length, truncation=True)
            if trg_feature is not None and trg_feature in records:
                assert cls_name in self.cls_heads
                model_inputs['labels'] = [self.config.label2id[cls_name][x]
                                          for x in records[trg_feature]]
            return model_inputs

        return datasets.map(tokenize_records, batched=True)

    def predict(self, testset, *, output_dir='.', bs=64, output_logits=False, temperature=1,
                log_level='passive', disable_tqdm=False, dataset_params={}):
        """Apply inference on test dataset, return predictions, labels (optionally) and probs."""
        testset = self.create_dataset(testset, **dataset_params)

        # create trainer with minimal setup for test-time
        trainer = self.get_trainer(
            output_dir=output_dir, bs=bs, log_level=log_level, disable_tqdm=disable_tqdm)

        # run inference
        logits, targs, metrics = self._predict(testset, trainer)
        if isinstance(logits, dict):
            probs = {k: numpy_utils.softmax(v, axis=-1, temperature=temperature)
                     for k, v in logits.items()}
            preds = {k: v.argmax(-1) for k, v in probs.items()}
        else:
            probs = numpy_utils.softmax(logits, axis=-1, temperature=temperature)
            preds = probs.argmax(-1)

        # include optional output values
        opt_kwargs = {}
        if output_logits:
            opt_kwargs['logits'] = logits

        return ClassificationOutput(
            predictions=preds, label_ids=targs, probs=probs, metrics=metrics, **opt_kwargs)
