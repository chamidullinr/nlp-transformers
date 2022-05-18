from transformers import (
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

__all__ = ["TrainingMixin", "Seq2seqTrainingMixin"]


class TrainingMixin:
    def get_trainer(
        self,
        output_dir,
        train_dataset=None,
        eval_dataset=None,
        *,
        no_epochs=1,
        bs=64,
        gradient_accumulation_steps=1,
        save_checkpoints=False,
        lr=2e-5,
        wd=0.01,
        lr_scheduler_type="linear",
        fp16=False,
        compute_metrics_cb=None,
        num_workers=0,
        resume_from_checkpoint=None,
        metric_for_best_model=None,
        greater_is_better=None,
        seed=42,
        log_level="passive",
        disable_tqdm=False,
        **kwargs
    ):
        """
        Get the trainer object for training and evaluating the network
        for given training and validation datasets.
        """
        assert hasattr(self, "model"), 'This object does not have "model" attribute.'
        assert hasattr(
            self, "tokenizer"
        ), 'This object does not have "tokenizer" attribute.'

        if eval_dataset is not None:
            evaluation_strategy = "epoch"
        else:
            evaluation_strategy = "no"

        if save_checkpoints:
            save_strategy = "epoch"
            load_best_model_at_end = True
        else:
            save_strategy = "no"
            load_best_model_at_end = False

        # define training arguments
        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=1,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=no_epochs,
            learning_rate=lr,
            weight_decay=wd,
            lr_scheduler_type=lr_scheduler_type,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            fp16=fp16,  # fp16 is not optimized on the current AWS GPU instance
            dataloader_num_workers=num_workers,
            resume_from_checkpoint=resume_from_checkpoint,
            seed=seed,
            log_level=log_level,
            disable_tqdm=disable_tqdm,
            report_to="none",
            **kwargs
        )

        # create data collator for splitting the data into batches
        data_collator = DataCollatorWithPadding(self.tokenizer)

        # create trainer instance
        trainer = Trainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_cb,
        )

        return trainer


class Seq2seqTrainingMixin:
    def get_trainer(
        self,
        output_dir,
        train_dataset=None,
        eval_dataset=None,
        *,
        no_epochs=1,
        bs=64,
        gradient_accumulation_steps=1,
        save_checkpoints=False,
        lr=2e-5,
        wd=0.01,
        lr_scheduler_type="linear",
        fp16=False,
        compute_metrics_cb=None,
        num_workers=0,
        resume_from_checkpoint=None,
        metric_for_best_model=None,
        greater_is_better=None,
        seed=42,
        log_level="passive",
        disable_tqdm=False,
        **kwargs
    ):
        """
        Get the trainer object for training and evaluating the network
        for given training and validation datasets.
        """
        assert hasattr(self, "model"), 'This object does not have "model" attribute.'
        assert hasattr(
            self, "tokenizer"
        ), 'This object does not have "tokenizer" attribute.'

        if eval_dataset is not None:
            evaluation_strategy = "epoch"
        else:
            evaluation_strategy = "no"

        if save_checkpoints:
            save_strategy = "epoch"
            load_best_model_at_end = True
        else:
            save_strategy = "no"
            load_best_model_at_end = False

        # define training arguments
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=1,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=no_epochs,
            learning_rate=lr,
            weight_decay=wd,
            lr_scheduler_type=lr_scheduler_type,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            fp16=fp16,  # fp16 is not optimized on the current AWS GPU instance
            dataloader_num_workers=num_workers,
            resume_from_checkpoint=resume_from_checkpoint,
            seed=seed,
            log_level=log_level,
            disable_tqdm=disable_tqdm,
            report_to="none",
            predict_with_generate=True,
            **kwargs
        )

        # create data collator for splitting the data into batches
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # create trainer instance
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_cb,
        )

        return trainer
