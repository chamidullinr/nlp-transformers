import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

from transformers import DistilBertPreTrainedModel, DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class FCSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self._init_weights(self.pre_classifier)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, pooled_output, labels=None):
        # apply forward pass
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        # evaluate loss
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return logits, loss


class DistilBertForMultiheadSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config, cls_heads):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.cls_heads = cls_heads
        self._cls_mode = 'all'

        self.distilbert = DistilBertModel(config)
        self.heads = nn.ModuleDict({name: FCSequenceClassification(config, num_labels)
                                    for name, num_labels in cls_heads.items()})

    def set_cls_mode(self, cls_mode):
        if cls_mode != 'all' and cls_mode not in self.cls_heads:
            raise ValueError(f'Unknown value "{self._cls_mode}" of argument "cls_mode". '
                             f'Allowed values are: {["all"] + list(self.cls_heads.keys())}')

        self._cls_mode = cls_mode

    @property
    def cls_mode(self):
        return self._cls_mode

    @cls_mode.setter
    def cls_mode(self, cls_mode):
        self.set_cls_mode(cls_mode)

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # apply transformer encoder
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)

        # apply classification head(s)
        if self._cls_mode == 'all':
            logits, loss = {}, {}
            for name, head in self.heads.items():
                logits[name], loss[name] = head(pooled_output, labels)
        elif self._cls_mode in self.heads:
            head = self.heads[self._cls_mode]
            logits, loss = head(pooled_output, labels)
        else:
            raise ValueError(f'Unknown value "{self._cls_mode}" of argument "cls_mode". '
                             f'Allowed values are: {["all"] + list(self.heads.keys())}')

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
