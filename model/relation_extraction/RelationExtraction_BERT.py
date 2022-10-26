import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from model.relation_extraction.components import WindowAttention, ScaledDotProductAttention
from pretrained_encoder.BERT.modeling.BERT import BertModel, BERTLayerNorm

class RelationExtraction_BERT(nn.Module):
    def __init__(self, config, window_size, layer_num):
        super(RelationExtraction_BERT, self).__init__()
        self.bert = BertModel(config)
        self.window_size = window_size
        self.layer_num = layer_num

        self.type_emb = nn.Embedding(6, config.hidden_size)
        self.fc_type2query = nn.Linear(2 * config.hidden_size, config.hidden_size)

        self.fc_pair = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.fc_content = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        self.classifier = nn.Linear(2 * config.hidden_size, 36)
        self.layer_norm = torch.nn.LayerNorm(2 * config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

        if self.layer_num > 0:
            self.window_attention = WindowAttention(encoder_type="BERT", config=config, window_size=self.window_size)
        self.type_attention = ScaledDotProductAttention(temperature=config.hidden_size ** 0.5)


    def forward(self, input_ids, token_type_ids, attention_mask, subj_mask, obj_mask, pair_entitytype_id, labels=None):
        batch_size, seq_length = input_ids.size(0), input_ids.size(1)
        sequence_output, _ = self.bert(input_ids.view(-1, seq_length), token_type_ids.view(-1, seq_length), attention_mask.view(-1, seq_length))

        if self.layer_num > 0:
            for i in range(self.layer_num):
                sequence_output = self.window_attention(sequence_output)

        type_embs = self.type_emb(torch.tensor([i for i in range(6)]).cuda()).unsqueeze(0).expand(batch_size, -1, -1)
        subj_typeemb = torch.bmm(pair_entitytype_id[:, 0, :].unsqueeze(1), type_embs).squeeze(1)
        obj_typeemb = torch.bmm(pair_entitytype_id[:, 1, :].unsqueeze(1), type_embs).squeeze(1)
        pair_typeemb = torch.cat((subj_typeemb, obj_typeemb), dim=-1)
        pair_type2query = self.dropout(self.fc_type2query(pair_typeemb))


        subj_token_mask = torch.where(subj_mask == -1, torch.ones_like(subj_mask), torch.zeros_like(subj_mask))
        obj_token_mask = torch.where(obj_mask == -1, torch.ones_like(obj_mask), torch.zeros_like(obj_mask))

        subj_mask = torch.where(subj_mask > 0, torch.ones_like(subj_mask), torch.zeros_like(subj_mask))
        obj_mask = torch.where(obj_mask > 0, torch.ones_like(obj_mask), torch.zeros_like(obj_mask))
        subj_features, _ = self.type_attention(pair_type2query.unsqueeze(1), sequence_output, sequence_output, subj_mask.unsqueeze(1))
        obj_features, _ = self.type_attention(pair_type2query.unsqueeze(1), sequence_output, sequence_output, obj_mask.unsqueeze(1))
        subj_features, obj_features = subj_features.squeeze(1), obj_features.squeeze(1)

        subj_content_feature = torch.bmm(subj_token_mask.unsqueeze(1), sequence_output).squeeze(1)
        obj_content_feature = torch.bmm(obj_token_mask.unsqueeze(1), sequence_output).squeeze(1)

        pair_output = torch.cat((subj_features, obj_features), dim=-1)
        pair_output = self.dropout(self.activation(self.fc_pair(pair_output)))

        content_output = torch.cat((subj_content_feature, obj_content_feature), dim=-1)
        content_output = self.dropout(self.activation(self.fc_content(content_output)))

        output = torch.cat((pair_output, content_output), dim=-1)

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