#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from mmbt.utils.utils import truncate_seq_pair, numpy_seed
import pickle

# pickleを保存
def save_pickle(obj, path):
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)


# pickleをロード
def load_pickle(path):
    with open(path, mode="rb") as f:
        obj = pickle.load(f)
    return obj


class JsonlDataset(Dataset):
    def __init__(self, df, tokenizer, transforms, vocab, args):

        self.data = df
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        self.max_seq_len -= args.num_image_embeds
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = (
            self.text_start_token
            + self.tokenizer(self.data.iloc[index]["text"])[
                : (self.args.max_seq_len - 1)
            ]
        )

        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        label = torch.LongTensor(
            [self.args.labels.index(self.data.iloc[index]["label"])]
        )

        image = None
        if self.data.iloc[index]["img"]:
            image = Image.open(os.path.join(self.data.iloc[index]["img"])).convert(
                "RGB"
            )
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))

        image = self.transforms(image)
        key = self.data.iloc[index]["key"] * torch.ones(1)
        # The first SEP is part of Image Token.
        segment = segment[1:]
        sentence = sentence[1:]
        # The first segment (0) is of images.
        segment += 1

        return key, sentence, segment, image, label


class JsonlDataset_for_production(Dataset):
    def __init__(self, df, tokenizer, transforms, vocab, args):
        self.data = df
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.text_start_token = ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        self.max_seq_len -= args.num_image_embeds
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = (
            self.text_start_token
            + self.tokenizer(self.data.iloc[index]["text"])[
                : (self.args.max_seq_len - 1)
            ]
        )

        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        image = None
        if self.data.iloc[index]["img"]:
            image = Image.open(os.path.join(self.data.iloc[index]["img"])).convert(
                "RGB"
            )
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))

        image = self.transforms(image)

        # The first SEP is part of Image Token.
        segment = segment[1:]
        sentence = sentence[1:]
        # The first segment (0) is of images.
        segment += 1

        label = torch.ones(1)  # ダミー

        return index, sentence, segment, image, label
