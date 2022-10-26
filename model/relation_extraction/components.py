import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class WindowAttention(nn.Module):
    def __init__(self, encoder_type, config, window_size):
        super(WindowAttention, self).__init__()
        self.encoder_type = encoder_type

        self.window_size = window_size
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.attention = MultiHeadAttention(config.num_attention_heads, config.hidden_size, self.attention_head_size, self.attention_head_size, config.attention_probs_dropout_prob)


    def forward(self, sequence_output):
        batch_size, seq_length = sequence_output.size(0), sequence_output.size(1)
        sequence_output, _ = self.attention(self.query(sequence_output), self.key(sequence_output), sequence_output, self.window_mask(batch_size, seq_length))
        return sequence_output

    def window_mask(self, batch_size, seq_length):
        maxidx = seq_length - 5 if self.encoder_type == "BERT" else seq_length - 6
        length = int(self.window_size/2)
        window_mask = torch.zeros((seq_length, seq_length))
        window_mask = window_mask.cuda()
        for i in range(0, seq_length):
            if i >= 1 and i <= maxidx:
                start = max(i - length, 1)
                end = min(i + length, maxidx)
                for j in range(start, end+1):
                    window_mask[i][j] = 1
            else:
                window_mask[i][i] = 1
        window_mask = window_mask.unsqueeze(0).expand(batch_size, -1, -1)
        return window_mask


