import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss

from pretrained_encoder.RoBERTa.modeling.RoBERTa import RobertaPreTrainedModel, RobertaModel


class EntityTyping(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, activation='tanh'):
        super().__init__(config)
        self.roberta = RobertaModel(config)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 5)



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        seq_length = input_ids.size(1)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        feature = torch.bmm(mention_mask.unsqueeze(1), sequence_output)
        logits = self.classifier(feature)
        logits = logits.view(-1, 5)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 5)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits