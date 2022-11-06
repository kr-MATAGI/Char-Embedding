import torch
import torch.nn as nn
import torch.optim as optim
import re

import copy
import numpy as np
import math
import pickle
from data_def import Sentence, NE, Word, Morp

from char_CNN import CharCNN

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import List, Dict
from model import CharELMo
from tqdm import tqdm

from jamo import h2j, j2hcj
from string import ascii_lowercase

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
def split_hangul_components(all_hangul: List[str]):
#======================================================
    comp_list = []

    # 한글
    for char in all_hangul:
        ch_split = list(j2hcj(h2j(char)))
        comp_list.extend(ch_split)
    comp_list = list(set(comp_list))

    # 숫자
    for num in range(0, 10):
        comp_list.append(str(num))

    # 영어
    comp_list.extend(ascii_lowercase)

    # TODO: 특수 문자

    comp_list = sorted(comp_list)
    comp_list.insert(0, "<p>") # PAD
    comp_list.insert(1, "<u>") # UNK
    comp_list.insert(2, "<e>") # end of char
    comp_list.insert(3, "_") # space

    return comp_list

#======================================================
def make_jamo_one_hot(vocab_dict: Dict[str, int], vocab_size: int, sent: str = ""):
#======================================================
    split_by_char = list(sent)

    chars_one_hot = np.zeros((128 * 3, vocab_size))
    add_idx = 0
    for ch_idx, char in enumerate(split_by_char):
        if 128 <= ch_idx:
            break
        jamo = j2hcj(h2j(char))
        zeros_np = np.zeros((3, vocab_size))
        for jm_idx, jm in enumerate(jamo):
            if jm not in char_dic.keys():
                zeros_np[jm_idx, char_dic["<u>"]] = 1
            else:
                zeros_np[jm_idx, char_dic[jm]] = 1
        for jm_idx in range(3):
            chars_one_hot[add_idx] = zeros_np[jm_idx]
            add_idx += 1

    return chars_one_hot


if "__main__" == __name__:
    all_hangul = []
    add_ch = ''
    while True:
        add_ch = kor_letter_from(add_ch)
        if add_ch is False:
            break
        all_hangul.append(add_ch)
    hangul_comp_list = split_hangul_components(all_hangul)
    char_dic = {c: i for i, c in enumerate(hangul_comp_list)}
    vocab_size = len(char_dic)
    print("Char Dict: ", char_dic)
    print("Vocab Size: ", vocab_size)

    train_sents, dev_sents, test_sents = load_sentences_datasets("./NIKL_ne_parsed.pkl")
    total_sents = train_sents + dev_sents + test_sents  # List

    jamo_inputs = []
    for sent in dev_sents:
        jamo_one_hot = make_jamo_one_hot(vocab_dict=char_dic, vocab_size=vocab_size, sent=sent)
        jamo_inputs.append(jamo_one_hot)
        break
    jamo_inputs = torch.FloatTensor(jamo_inputs) # [batch_siz, seq_len, 초/중/종성, vocab_size]
    model = CharCNN(vocab_dict=char_dic, vocab_size=vocab_size, seq_len=128)
    outputs = model(jamo_inputs)

    print("[run_cnn.py] __main__")