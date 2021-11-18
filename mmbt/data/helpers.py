#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root diftory of this source tree.
#

import functools
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from mmbt.data.dataset import JsonlDataset
from mmbt.data.dataset import JsonlDataset_for_production
from mmbt.data.vocab import Vocab


# 画像に対する前処理方法を定義
def get_transforms(args):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


# ラベル毎の出現回数を取得
def get_labels_and_frequencies(data_dict):
    label_freqs = Counter()
    df_train = data_dict["train"]
    data_labels = df_train.label.tolist()
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return sorted(list(label_freqs.keys())), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab


def collate_fn(batch, args):

    lens = [len(row[1]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()
    # print([row[0] for row in batch])
    index_tensor = torch.tensor([row[0] for row in batch])
    img_tensor = None
    img_tensor = torch.stack([row[3] for row in batch])
    # Single Label case
    tgt_tensor = torch.cat([row[4] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[1:3]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    return (
        index_tensor,
        text_tensor,
        segment_tensor,
        mask_tensor,
        img_tensor,
        tgt_tensor,
    )


# data_dictからdataloaders_dictを生成
def get_data_loaders(args, data_dict):
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    ).tokenize
    # 画像前処理用のtransform
    transforms = get_transforms(args)

    args.labels, args.label_freqs = get_labels_and_frequencies(data_dict)
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train = JsonlDataset(data_dict["train"], tokenizer, transforms, vocab, args,)

    args.train_data_len = len(train)

    dev = JsonlDataset(data_dict["val"], tokenizer, transforms, vocab, args,)

    test = JsonlDataset(data_dict["test"], tokenizer, transforms, vocab, args,)

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    test_loader = DataLoader(
        test,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader, "test": test_loader}

    return dataloaders_dict


# data_dictからdataloaders_dictを生成
def get_data_loaders_for_production(args, data_dict):
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    ).tokenize
    # 画像前処理用のtransform
    transforms = get_transforms(args)
    collate = functools.partial(collate_fn, args=args)
    # args.labels = [0, 1]  # 明示的に記述せず自動取得できるようにするのが望ましい
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    # args.n_classes = len(args.labels)

    production = JsonlDataset_for_production(
        data_dict["production"], tokenizer, transforms, vocab, args,
    )
    production_loader = DataLoader(
        production,
        batch_size=1,  # 1行ずつデータを取り出して予測するため 1 としている
        shuffle=False,  # データの順番を変えると元データと結合できなくなるためFalseにしている
        num_workers=args.n_workers,
        collate_fn=collate,
    )
    dataloaders_dict = {"production": production_loader}

    return dataloaders_dict
