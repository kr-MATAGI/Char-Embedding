import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#====================================================
class CharELMo(nn.Module):
#====================================================
    def __init__(self,
                 vocab_size: int,
                 embed_size: int = 512,
                 hidden_size: int = 768,
                 dropout_rate: float = 0.1,
                 max_seq_len: int = 128
                 ):
        super(CharELMo, self).__init__()

        # Init
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Char Embedding
        # 문자당 임베딩 생성
        self.embedding = nn.Embedding(vocab_size, embed_size) # [batch, seq_len, embed_dim]

        # biLM
        self.forward_lm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                                  batch_first=True, bidirectional=False, num_layers=2)

        self.backward_lm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                                   batch_first=True, bidirectional=False, num_layers=2)

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Softmax + FFN
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor):
        char_embed = self.embedding(x) # [batch_size, seq_len(char)]

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