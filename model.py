import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#====================================================
class Highway(nn.Module):
#====================================================
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear
        return x

#====================================================
class CharELMo(nn.Module):
#====================================================
    def __init__(self,
                 vocab_size: int,
                 embed_size: int = 2048,
                 hidden_size: int = 4096,
                 dropout_rate: float = 0.1,
                 max_seq_len: int = 128
                 ):
        super(CharELMo, self).__init__()

        # Init
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Char Embedding
        # 문자 단위로 이루어진 문장
        self.embedding = nn.Embedding(vocab_size, embed_size) # [batch, seq_len, embed_dim]

        # Highway Netwrok
        self.highway = Highway(num_layers=2, size=embed_size, f=nn.ReLU())

        # biLM
        self.forward_lm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size//2,
                                  batch_first=True, bidirectional=True, num_layers=2)

        self.backward_lm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size//2,
                                   batch_first=True, bidirectional=True, num_layers=2)

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Softmax + FFN
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor):
        char_embed = self.embedding(x) # [batch_size, seq_len(char)]
        char_embed = self.highway(char_embed)

        reverse_char_embed = char_embed.flip(dims=[0, 1])
        input_len, sorted_idx = seq_len.sort(0, descending=True)
        input_len = input_len.squeeze(1)
        # input_seq2idx = seq_len[sorted_idx]

        packed_char_emb = pack_padded_sequence(char_embed, input_len.tolist(), batch_first=True)
        packed_reverse_char_embed = pack_padded_sequence(reverse_char_embed, input_len.tolist(), batch_first=True)

        # BiLSTM - 1
        f_lm_out, f_h = self.forward_lm(packed_char_emb)
        f_lm_out, f_lm_out_len = pad_packed_sequence(f_lm_out, batch_first=True,
                                                     padding_value=0, total_length=self.max_seq_len)
        f_lm_out = self.dropout(f_lm_out)

        b_lm_out, b_h = self.backward_lm(packed_reverse_char_embed)
        b_lm_out, b_lm_out_len = pad_packed_sequence(b_lm_out, batch_first=True,
                                                     padding_value=0, total_length=self.max_seq_len)
        b_lm_out = self.dropout(b_lm_out)
        # concat_out = torch.concat([char_embed, f_lm_out, b_lm_out], -1)
        concat_out = torch.concat([f_lm_out, b_lm_out], -1)

        # Softmax + FFN
        logits = self.classifier(concat_out)
        logits = F.softmax(logits, -1)

        return logits