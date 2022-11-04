import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np
import math
import pickle
from data_def import Sentence, NE, Word, Morp

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import List, Dict
from model import CharELMo
from tqdm import tqdm


from ELMo import ELMo

### GLOBAL
best_ppl = math.inf
ppl_scores = [] # (epoch_, ppl)

#======================================================
class ELMoDatasets(Dataset):
#======================================================
    def __init__(self,
                 sentences: List[str],
                 char_dic: Dict[str, int],
                 vocab_size: int,
                 seq_len: int = 128,
                 ):
        self.sentences = sentences
        self.char_dic = char_dic
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        X = [char_dic["[CLS]"]]
        y = []
        valid_seq_len = []
        for c_idx in range(0, len(self.sentences[idx]) - 1):
            X.append(char_dic[self.sentences[idx][c_idx]])
            y.append(char_dic[self.sentences[idx][c_idx + 1]])
        X.append(char_dic["[SEP]"])
        y.append(char_dic["[SEP]"])
        # y_reverse = list(reversed(y))
        valid_seq_len.append(len(X))
        valid_seq_len = [l if l < self.seq_len else self.seq_len for l in valid_seq_len]

        if len(X) >= self.seq_len:
            X = X[:self.seq_len]
        else:
            X += [0] * (self.seq_len - len(X))
        if len(y) >= self.seq_len:
            y = y[:self.seq_len]
            # y_reverse = y_reverse[:self.seq_len]
        else:
            y += [0] * (self.seq_len - len(y))
            # y_reverse += [0] * (self.seq_len - len(y_reverse))
        X = torch.LongTensor(X)
        y = torch.LongTensor(y)
        # y_reverse = torch.LongTensor(y_reverse)
        valid_seq_len = torch.IntTensor(valid_seq_len)
        return X, y, valid_seq_len

#======================================================
def kor_letter_from(letter):
#=========================================b=============
    lastLetterInt = 15572643

    if not letter:
        return '가'

    a = letter
    b = a.encode('utf8')
    c = int(b.hex(), 16)

    if c == lastLetterInt:
        return False

    d = hex(c + 1)
    e = bytearray.fromhex(d[2:])

    flag = True
    while flag:
        try:
            r = e.decode('utf-8')
            flag = False
        except UnicodeDecodeError:
            c = c + 1
            d = hex(c)
            e = bytearray.fromhex(d[2:])

    return e.decode()

#======================================================
def make_char_dict(sentences: List[str]):
#======================================================
    char_set = []
    for sent in sentences:
        char_set.extend(list(sent.replace(" ", "_").split(" ")[0]))

    # 한국어 모든 글자 추가
    add_ch = ''
    while True:
        add_ch = kor_letter_from(add_ch)
        if add_ch is False:
            break
        char_set.append(add_ch)

    char_set = list(set(char_set))
    char_set.insert(1, "[CLS]")
    char_set.insert(2, "[SEP]")
    char_set.insert(3, "[UNK]")
    char_set.insert(0, "[PAD]")
    char_dic = {c: i for i, c in enumerate(char_set)}
    vocab_size = len(char_dic)
    # print("char_dic: ", char_dic)

    return char_set, char_dic, vocab_size

#======================================================
def load_sentences_datasets(src_path: str) -> List[str]:
#======================================================
    # Load
    load_src_datasets: List[Sentence] = []
    with open(src_path, mode="rb") as src_file:
        load_src_datasets = pickle.load(src_file)
        print(f"[load_sentences_datasets] Load data size: {len(load_src_datasets)}")

    total_sents = []
    for sent in load_src_datasets:
        total_sents.append(sent.text.replace(" ", "_"))
    split_size = int(len(total_sents) * 0.1)
    train_idx = split_size * 7
    dev_idx = train_idx + split_size

    train_sents = total_sents[:train_idx]
    dev_sents = total_sents[train_idx:dev_idx]
    test_sents = total_sents[dev_idx:]
    print(f"[load_sentences_datasets] size - train: {len(train_sents)}, dev: {len(dev_sents)}, test: {len(test_sents)}")

    return train_sents, dev_sents, test_sents

#======================================================
def evaluate(model, eval_datasets, device, batch_size: int = 128):
#======================================================
    eval_loss = 0.0
    nb_eval_steps = 0

    eval_sampler = SequentialSampler(eval_datasets)
    eval_dataloader = DataLoader(eval_datasets, sampler=eval_sampler, batch_size=batch_size)
    eval_pbar = tqdm(eval_dataloader)
    criterion = nn.CrossEntropyLoss()
    for batch in eval_pbar:
        model.eval()
        with torch.no_grad():
            X, y, valid_seq_len = batch
            X = X.to(device)
            y = y.to(device)
            valid_seq_len = valid_seq_len.to(device)
            outputs = model(X, valid_seq_len)
            loss = criterion(outputs.view(-1, vocab_size), y.view(-1))
            eval_loss += loss.mean().item()
            nb_eval_steps += 1
            perplexity = torch.exp(loss).item()
            eval_pbar.set_description("Eval Loss - %.04f, PPL: %.04f" % ((eval_loss / nb_eval_steps), perplexity))

            results = outputs.argmax(dim=-1)
            for r_idx, res in enumerate(results):
                predict_str = "".join([char_set[x] for x in res])
                print(f"{r_idx}: \n {predict_str}")

### Main ###
if "__main__" == __name__:
    print("[main.py] __main__")

    # Datasets
    train_sents, dev_sents, test_sents = load_sentences_datasets("./NIKL_ne_parsed.pkl")
    total_sents = train_sents + dev_sents + test_sents # List
    char_set, char_dic, vocab_size = make_char_dict(total_sents)
    # x_one_hot = [np.eye(vocab_size)[x] for x in x_data]  # x 데이터는 원-핫 인코딩
    print("Vocab Size:", vocab_size)

    train_datasets = ELMoDatasets(train_sents, char_dic, vocab_size)
    dev_datasets = ELMoDatasets(dev_sents, char_dic, vocab_size)

    # Config
    total_epoch = 20
    learning_rate = 1e-3
    # filters = [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[__main__] Device:", device, torch.cuda.is_available())

    model = CharELMo(vocab_size=vocab_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # Train
    train_loss = 0.0
    train_step = 0
    train_sampler = RandomSampler(train_datasets)
    model.zero_grad()
    for epoch in range(total_epoch):
        model.train()
        train_data_loader = DataLoader(train_datasets, sampler=train_sampler, batch_size=128)
        train_pbar = tqdm(train_data_loader)
        for batch_idx, samples in enumerate(train_pbar):
            model.train()
            X, y, valid_seq_len = samples
            X = X.to(device)
            y = y.to(device)
            valid_seq_len = valid_seq_len.to(device)
            outputs = model(X, valid_seq_len)
            loss = criterion(outputs.view(-1, vocab_size), y.view(-1))
            loss.backward()
            train_loss += loss.item()
            train_step += 1
            optimizer.step()

            train_pbar.set_description("Train Loss - %.04f" % (train_loss / train_step))

        # Eval
        evaluate(model, dev_datasets, device, batch_size=128)