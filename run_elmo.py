import torch
import torch.nn as nn
import torch.optim as optim
import re
import os

import copy
import numpy as np
import math
import pickle
from data_def import Sentence, NE, Word, Morp

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import List, Dict
from model import CharELMo
from tqdm import tqdm
import copy

### GLOBAL
g_best_ppl = math.inf
g_best_loss = math.inf
g_best_epoch = -1
g_test_mode = False

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
        X = []
        y = []
        for c_idx in range(0, len(self.sentences[idx]) - 1):
            X.append(char_dic[self.sentences[idx][c_idx]])
            y.append(char_dic[self.sentences[idx][c_idx+1]])

        y.append(char_dic["[BOS]"])
        y_reverse = list(reversed(y))

        if len(X) >= self.seq_len:
            X = X[:self.seq_len]
        else:
            X += [0] * (self.seq_len - len(X))

        if len(y) >= self.seq_len:
            y = y[:self.seq_len]
            y_reverse = y_reverse[:self.seq_len]
        else:
            y += [0] * (self.seq_len - len(y))
            y_reverse += [0] * (self.seq_len - len(y_reverse))

        X = torch.LongTensor(X)
        y = torch.LongTensor(y)
        y_reverse = torch.LongTensor(y_reverse)

        return X, y, y_reverse

#======================================================
def kor_letter_from(letter):
#======================================================
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

    char_set = sorted(list(set(char_set)))
    # char_set.insert(1, "[CLS]")
    # char_set.insert(2, "[SEP]")
    char_set.insert(2, "[UNK]")
    char_set.insert(1, "[BOS]")
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
        if 10 <= len(sent.text):
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
def evaluate(model, eval_datasets, device, epoch, batch_size: int = 128):
#======================================================
    eval_loss = 0.0
    nb_eval_steps = 0
    perplexity = 0.0

    eval_sampler = SequentialSampler(eval_datasets)
    eval_dataloader = DataLoader(eval_datasets, sampler=eval_sampler, batch_size=batch_size)
    eval_pbar = tqdm(eval_dataloader)
    criterion = nn.CrossEntropyLoss()
    for batch in eval_pbar:
        model.eval()
        with torch.no_grad():
            X, y, y_reverse, valid_seq_len = batch
            X = X.to(device)
            y = y.to(device)
            y_reverse = y_reverse.to(device)
            valid_seq_len = valid_seq_len.to(device)
            f_logits, b_logits = model(X, valid_seq_len)
            # f_logits = model(X, valid_seq_len)
            f_loss = criterion(f_logits.view(-1, vocab_size), y.view(-1))
            b_loss = criterion(b_logits.view(-1, vocab_size), y_reverse.view(-1))
            eval_loss += f_loss.mean().item() + b_loss.mean().item()
            nb_eval_steps += 1
            perplexity = torch.exp(f_loss + b_loss).item()
            # perplexity = torch.exp(f_loss).item()
            eval_pbar.set_description("Eval Loss - %.04f, PPL: %.04f" % ((eval_loss / nb_eval_steps), perplexity))

            results = f_logits.argmax(dim=-1)
            for r_idx, res in enumerate(results):
                predict_str = "".join([char_set[x] for x in res])
                print(f"{r_idx}: \n {predict_str}")

        # Write
        global g_test_mode
        if g_test_mode:
            with open("./result_"+str(epoch)+".txt", mode="w") as test_res_file:
                test_res_file.write("Epoch: "+str(epoch)+" : "+"Loss : "+str(eval_loss / nb_eval_steps) +
                                    "PPL : "+str(perplexity)+"\n")

    global g_best_loss, g_best_ppl, g_best_epoch
    if g_best_loss > eval_loss / nb_eval_steps:
        g_best_epoch = epoch
        g_best_loss = eval_loss / nb_eval_steps
        g_best_ppl = perplexity

### Main ###
if "__main__" == __name__:
    print("[run_elmo.py] __main__")

    # Datasets
    train_sents, dev_sents, test_sents = load_sentences_datasets("./NIKL_ne_parsed.pkl")
    total_sents = train_sents + dev_sents + test_sents # List
    char_set, char_dic, vocab_size = make_char_dict(total_sents)
    print("Vocab Size:", vocab_size)
    # Save Char Vocab
    with open("./char_elmo_vocab.pkl", mode="wb") as vocab_pkl:
        pickle.dump(char_dic, vocab_pkl)
        print(f"[run_elmo.py][__main__] Vocab Saved ! - len: {len(char_dic)}")

    train_datasets = ELMoDatasets(train_sents, char_dic, vocab_size)
    dev_datasets = ELMoDatasets(dev_sents, char_dic, vocab_size)
    test_datasets = ELMoDatasets(test_sents, char_dic, vocab_size)

    # Config
    total_epoch = 20
    learning_rate = 1e-5
    # filters = [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[__main__] Device:", device, torch.cuda.is_available())

    if g_test_mode:
        root_model_path = "./model"
        saved_models = sorted(os.listdir(root_model_path))
        print(f"model_list: {saved_models}")
        for model_path in saved_models:
            print(f"[__main__][test_mode] {model_path}")
            test_model_epoch = int(model_path.split("_")[0])
            model = CharELMo(vocab_size=vocab_size)
            model.load_state_dict(torch.load(root_model_path+"/"+model_path, map_location=device))
            model.to(device)
            evaluate(model, eval_datasets=test_datasets,
                     device=device, epoch=test_model_epoch, batch_size=128)
            print(f"[__main__][test_mode] TEST END - {model_path}")
    else:
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
                X, y, y_reverse = samples
                X = X.to(device)
                y = y.to(device)
                y_reverse = y_reverse.to(device)
                f_logits, b_logits = model(X)
                # f_logits = model(X, valid_seq_len)
                f_loss = criterion(f_logits.view(-1, vocab_size), y.view(-1))
                b_loss = criterion(b_logits.view(-1, vocab_size), y_reverse.view(-1))
                total_loss = f_loss + b_loss
                total_loss.backward()
                train_loss += total_loss.item()
                train_step += 1
                optimizer.step()

                train_pbar.set_description("Train Loss - %.04f" % (train_loss / train_step))
            # end loop, batch

            # Eval
            evaluate(model, test_datasets, device, epoch=epoch+1, batch_size=128)

            # Save Model
            torch.save(model.state_dict(), "./model/" + str(epoch + 1) + "_model.pth")

        # Save Best Model
        print("Best Model epoch: ", g_best_epoch, "loss: ", g_best_loss, "ppl: ", g_best_ppl)
        # torch.save(g_best_model, "./best_model.pth")