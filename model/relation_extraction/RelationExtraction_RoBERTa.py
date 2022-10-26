import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

from model.relation_extraction.components import WindowAttention, ScaledDotProductAttention
from pretrained_encoder.RoBERTa.modeling.RoBERTa import RobertaPreTrainedModel, RobertaModel


class RelationExtraction_RoBERTa(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, layer_num, window_size, activation='tanh'):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.layer_num = layer_num
        self.window_size = window_size

        self.type_emb = nn.Embedding(6, config.hidden_size)
        self.fc_type2query = nn.Linear(2 * config.hidden_size, config.hidden_size)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, 36)

        self.fc_pair_content = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.fc_pair_entitytype = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = torch.nn.LayerNorm(2 * config.hidden_size)

        self.init_weights()

        self.window_attention = WindowAttention("RoBERTa", config, self.window_size)
        self.attention = ScaledDotProductAttention(temperature=config.hidden_size ** 0.5)



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            subj_mask=None,
            obj_mask=None,
            pair_entitytype_id=None,
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

        for i in range(self.layer_num):
            sequence_output = self.window_attention(sequence_output)

        type_embs = self.type_emb(torch.tensor([i for i in range(6)]).cuda()).unsqueeze(0).expand(sequence_output.shape[0], -1, -1)
        subj_typeemb = torch.bmm(pair_entitytype_id[:, 0, :].unsqueeze(1), type_embs).squeeze(1)
        obj_typeemb = torch.bmm(pair_entitytype_id[:, 1, :].unsqueeze(1), type_embs).squeeze(1)

        pair_typeemb = torch.cat((subj_typeemb, obj_typeemb), dim=-1)
        pair_type2query = self.dropout(self.fc_type2query(pair_typeemb))

        subj_token_mask = torch.where(subj_mask == -1, torch.ones_like(subj_mask), torch.zeros_like(subj_mask))
        obj_token_mask = torch.where(obj_mask == -1, torch.ones_like(obj_mask), torch.zeros_like(obj_mask))
        subj_mask = torch.where(subj_mask > 0, torch.ones_like(subj_mask), torch.zeros_like(subj_mask))
        obj_mask = torch.where(obj_mask > 0, torch.ones_like(obj_mask), torch.zeros_like(obj_mask))


        subj_features, _ = self.attention(pair_type2query.unsqueeze(1), sequence_output, sequence_output, subj_mask.unsqueeze(1))
        obj_features, _ = self.attention(pair_type2query.unsqueeze(1), sequence_output, sequence_output, obj_mask.unsqueeze(1))
        subj_features, obj_features = subj_features.squeeze(1), obj_features.squeeze(1)

        subj_token_feature = torch.bmm(subj_token_mask.unsqueeze(1), sequence_output).squeeze(1)
        obj_token_feature = torch.bmm(obj_token_mask.unsqueeze(1), sequence_output).squeeze(1)

        pair_content_output = torch.cat((subj_features, obj_features), dim=-1)
        pair_content_output = self.dropout(self.activation(self.fc_pair_content(pair_content_output)))
        pair_entitytype_output = torch.cat((subj_token_feature, obj_token_feature), dim=-1)
        pair_entitytype_output = self.dropout(self.activation(self.fc_pair_entitytype(pair_entitytype_output)))


        output = torch.cat((pair_content_output, pair_entitytype_output), dim=-1)

        output = self.layer_norm(output)
        output = self.dropout(output)

        logits = self.classifier(output)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 36)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits