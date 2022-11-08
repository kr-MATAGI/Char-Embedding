import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

'''
    pad_packed, pack_padded 쓰면 [PAD] 처리 못 함
'''
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
                 embed_size: int = 1024,
                 hidden_size: int = 768,
                 dropout_rate: float = 0.1,
                 max_seq_len: int = 128,
                 mode: str = "train" # or "repr"
                 ):
        super(CharELMo, self).__init__()

        # Init
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.mode = mode

        # Char Embedding
        # 문자 단위로 이루어진 문장
        self.embedding = nn.Embedding(vocab_size, embed_size) # [batch, seq_len, embed_dim]

        # BiLM
        self.forward_lm_1 = nn.LSTM(input_size=embed_size, hidden_size=hidden_size//2,
                                    batch_first=True, bidirectional=True, num_layers=1, dropout=self.dropout_rate)
        self.forward_lm_2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size//2,
                                    batch_first=True, bidirectional=True, num_layers=1, dropout=self.dropout_rate)

        self.backward_lm_1 = nn.LSTM(input_size=embed_size, hidden_size=hidden_size//2,
                                     batch_first=True, bidirectional=True, num_layers=1, dropout=self.dropout_rate)
        self.backward_lm_2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size//2,
                                     batch_first=True, bidirectional=True, num_layers=1, dropout=self.dropout_rate)

        # Softmax + FFN
        self.f_classifier = nn.Linear(hidden_size, vocab_size)
        self.b_classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor):
        char_embed = self.embedding(x) # [batch_size, seq_len(char)]
        reverse_char_embed = char_embed.flip(dims=[0, 1])

        # F_BiLSTM - 1
        f_lm_out_1, (f_h_1, f_c_1) = self.forward_lm_1(char_embed)

        # F_BiLSTM - 2
        f_lm_out_2, (f_h_2, f_c_2) = self.forward_lm_2(f_lm_out_1, (f_h_1, f_c_1))

        # B_BiLSTM - 1
        b_lm_out_1, (b_h_1, b_c_1) = self.backward_lm_1(reverse_char_embed)

        # B_BiLSTM - 2
        b_lm_out_2, (b_h_2, b_c_2) = self.backward_lm_2(b_lm_out_1, (b_h_1, b_c_1))

        if "train" == self.mode:
            # Softmax + FFN
            f_logits = self.f_classifier(f_lm_out_2) # [batch_size, seq_len, vocab_size]
            b_logits = self.b_classifier(b_lm_out_2)

            return f_logits, b_logits
        else:
            forward_concat = torch.concat([f_lm_out_1, b_lm_out_1], dim=-1)
            backward_concat = torch.concat([f_lm_out_2, b_lm_out_2], dim=-1)

            return char_embed, forward_concat, backward_concat

        # logits = self.classifier(concat_out)
        # return logits