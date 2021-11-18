#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from scripts.pytorch_pretrained_bert.modeling import BertModel

from mmbt.models.image import ImageEncoder
import torch.nn.functional as F
import numpy as np

import pickle


class ImageBertEmbeddings(nn.Module):
    # 1.CLSをベクトル化 2.SEPをベクトル化 3.画像をベクトル化 4.1~3をcat 4.3を標準化  5.4に対してdropout処理をする

    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        self.img_embeddings = nn.Linear(
            args.img_hidden_sz, args.hidden_sz
        )  # 画像ベクトルサイズ（2048次元）をbertの隠れ層（768次元）に圧縮
        self.position_embeddings = (
            embeddings.position_embeddings
        )  # 入力トークンの位置情報を把握するためのベクトル(最大トークン数の種類分のベクトル表現)
        self.token_type_embeddings = (
            embeddings.token_type_embeddings
        )  # 各単語が1文目なのか2文目なのかn文目なのかを示すベクトル
        self.word_embeddings = embeddings.word_embeddings  # 単語ベクトル
        self.LayerNorm = embeddings.LayerNorm  # 標準化させるやつ
        self.dropout = nn.Dropout(p=args.dropout)  # ドロップアウト

    def forward(self, input_imgs, token_type_ids):
        # print(('input_imgs',input_imgs)) # tensor([[[1.7565, 0.3654, 0.0763,  ..., 0.0407, 0.4960, 0.6700],
        # print(('input_imgs',input_imgs.size())) # torch.Size([32, 3, 2048])
        bsz = input_imgs.size(0)  # bszはバッチサイズ
        # print(('bsz',bsz)) # ('bsz', 32)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token
        # print(('self.args.num_image_embeds',self.args.num_image_embeds)) # 3
        # print(('seq_length',seq_length)) # 5

        # vocab.stoiは単語インデックス辞書
        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        # print(('cls_id',cls_id)) # ('cls_id', tensor([101], device='cuda:0'))
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        # print(('cls_id',cls_id)) # ('cls_id', tensor([[101],[101],・・・[101]],device='cuda:0')) 32

        # sepを単語ベクトル化
        cls_token_embeds = self.word_embeddings(cls_id)
        # print(('cls_token_embeds',cls_token_embeds)) # tensor([[[ 0.0136, -0.0265, -0.0235,  ...,
        # print('cls_token_embeds.shape',cls_token_embeds.shape) # torch.Size([32, 1, 768])
        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        # print(('sep_id',sep_id)) # ('sep_id', tensor([102], device='cuda:0'))
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        # print(('sep_id',sep_id)) # ('sep_id', tensor([[102],[102],・・・[102]],device='cuda:0')) 32

        # sepを単語ベクトル化
        sep_token_embeds = self.word_embeddings(sep_id)
        # print(('sep_token_embeds',sep_token_embeds)) # ('sep_token_embeds', tensor([[[-0.0145, -0.0100,  0.0060,  ..., -0.0250]], device='cuda:0', grad_fn=<EmbeddingBackward>))
        # print('sep_token_embeds.shape',sep_token_embeds.shape) # torch.Size([32, 1, 768])

        # 画像ベクトル2048次元を768次元に圧縮
        imgs_embeddings = self.img_embeddings(input_imgs)
        # print('imgs_embeddings',imgs_embeddings) # imgs_embeddings tensor([[[-0.2272, -0.0792, -0.2961,  ...,  0.4226, -0.1053,  0.5139],
        # print('imgs_embeddings.shape',imgs_embeddings.shape) # torch.Size([32, 3, 768])

        # cls+imgs+sep
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )
        # print('token_embeddings',token_embeddings) # tensor([[[ 0.0136, -0.0265, -0.0235,  ...,  0.0087,  0.0071,  0.0151],...]],device='cuda:0', grad_fn=<CatBackward>)
        # print('token_embeddings.shape',token_embeddings.shape) # torch.Size([32, 5, 768])
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        # print('position_ids',position_ids) # tensor([0, 1, 2, 3, 4], device='cuda:0')
        # print('position_ids.shape',position_ids.shape) # torch.Size([5])
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        # print('position_ids',position_ids) # tensor([[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],...]], device='cuda:0')
        # print('position_ids.shape',position_ids.shape) # torch.Size([32, 5])
        position_embeddings = self.position_embeddings(position_ids)
        # print('position_embeddings',position_embeddings) # tensor([[[ 1.7505e-02, -2.5631e-02, -3.6642e-02,  ...,  3.3437e-05,6.8312e-04,  1.5441e-02],
        # print('position_embeddings.shape',position_embeddings.shape) # torch.Size([32, 5, 768])
        # print('token_type_ids',token_type_ids) # tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],...[0, 0, 0, 0, 0]) # 0は0番目の文章という意味
        # print('token_type_ids',token_type_ids.shape) # torch.Size([32, 5])

        # 文章IDをベクトル化
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # print('token_type_embeddings',token_type_embeddings) # tensor([[[ 0.0004,  0.0110,  0.0037,  ..., -0.0066, -0.0034, -0.0086],
        # print('token_type_embeddings.shape',token_type_embeddings.shape) # torch.Size([32, 5, 768])

        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        # print('embeddings',embeddings) # tensor([[[ 0.0316, -0.0411, -0.0564,  ...,  0.0021,  0.0044,  0.0219],
        # print('embeddings.shape',embeddings.shape) # torch.Size([32, 5, 768])

        # 特徴量ごとに平均と分散を計算しデータの平均と分散をそれぞれ0および1にする
        embeddings = self.LayerNorm(embeddings)
        # print('embeddings(LayerNorm(embeddings))',token_embeddings) # tensor([[[ 0.0136, -0.0265, -0.0235,  ...,  0.0087,  0.0071,  0.0151],
        # print('embeddings(LayerNorm(embeddings)).shape',token_embeddings.shape) # torch.Size([32, 5, 768])
        # ドロップアウト
        embeddings = self.dropout(embeddings)
        # print('embeddings(dropout(embeddings))',token_embeddings) # tensor([[[ 0.0136, -0.0265, -0.0235,  ...,  0.0087,  0.0071,  0.0151],
        # print('embeddings(dropout(embeddings)).shape',token_embeddings.shape) # torch.Size([32, 5, 768])
        return embeddings


class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)  # bertモデル
        self.txt_embeddings = bert.embeddings  # 文章ベクトル

        self.img_embeddings = ImageBertEmbeddings(
            args, self.txt_embeddings
        )  # 画像ベクトル（bertっぽさをもたせたやつ）
        self.img_encoder = ImageEncoder(args)  # 画像情報を画像ベクトルに変換するエンコーダー
        self.encoder = bert.encoder  # 文章ベクトルに変換するエンコーダー

        self.pooler = bert.pooler  # encoderの後ろの、各タスクに接続する部分
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)  # クラス分類する全結合層

    def forward(self, input_txt, attention_mask, segment, input_img):
        """
        input_txt:単語IDのベクトル
        attention_mask:Tramsformerと同じ働きのマスキング
        segment:文章の区分となるベクトル。1文のみの場合はsegmentは全て0
        input_img:画像情報となるRGB3成分の3次元ベクトル
        """

        # 1.マスクの整形
        ## 1.1 マスクの次元数を「テキストのトークン数+画像の次元数(RGBの3次元)」に整える。つまり[batch_size,seq_length]の形
        ## 1.2 マスクを[batch_size,1,1,seq_length]の形にする
        ## 1.3 Attentionを掛けない部分はマイナス無限大にするための処理を加える

        # 2.画像ベクトルを生成
        ## 1.1 画像ベクトル[32, 3, 2048]を取得
        ## 1.2 整形するためのimgテンソルの箱（[32, 5]）作り
        ## 1.3 画像ベクトル[32, 3, 2048]を[32, 5, 768]に整形

        # 3.テキストベクトルを生成
        ## 2.1 単語ID化したテキストをbertで768次元に変換し、テキストベクトル[32, 24, 768]を取得

        # 3.画像＆テキストベクトルを生成
        ## 3.1 テキストベクトルと画像ベクトルをcat
        ## 3.2 マスク処理を加える

        # 4.画像＆テキストベクトルをpooling層に入れる

        bsz = input_txt.size(0)
        # batch_cnt = self.args.batch_cnt
        i_epoch = self.args.i_epoch
        # print(('batch_cnt',batch_cnt))
        # print(('bsz',bsz)) # ('bsz', 32)
        # attention_maskはinput_idが存在する箇所に1が立つ
        # print(('attention_mask',attention_mask)) # tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # print(('attention_mask',attention_mask.shape)) # torch.Size([32, 24]))

        # [32, 24]のattention_maskに[32, 5]のサイズ1のテンソルをcat（画像の次元とテキストの次元の箱に合わせる的な）
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.args.num_image_embeds + 2)
                .long()
                .cuda(),  # torch.Size([32, 5]))
                attention_mask,
            ],
            dim=1,
        )
        # print(('attention_mask(torch.cat)',attention_mask)) # tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0],
        # print(('attention_mask(torch.cat)',attention_mask.shape)) # torch.Size([32, 29])

        # マスクの変形。1番目と2番目の次元を1つ増やし、[batch_size,1,1,seq_length]の形にする
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # print(('extended_attention_mask',extended_attention_mask)) # tensor([[[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0]]],
        # print(('extended_attention_mask',extended_attention_mask.shape)) # torch.Size([32, 1, 1, 29])

        # torch.float32に型変換（次の処理でfloatが必要になる）
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        # print(('extended_attention_mask(dtype)',extended_attention_mask)) #  tensor([[[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]],
        # print(('extended_attention_mask(dtype)',extended_attention_mask.shape)) # torch.Size([32, 1, 1, 29])

        # Attentionを掛けない部分はマイナス無限大にしたいので、代わりに-10000を掛け算する（なぜこの処理をするのか？）
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print(('extended_attention_mask(引き算)',extended_attention_mask)) # tensor([[[[    -0.,     -0.,     -0.,     -0.,     -0.,     -0.,     -0.,-0.,     -0.,... -10000., -10000., -10000.,-10000.]]],
        # print(('extended_attention_mask(引き算)',extended_attention_mask.shape)) # torch.Size([32, 1, 1, 29])

        # [32, 5]のimgテンソルの箱作り。後に32*3*2048次元の画像ベクトルを32*5*768次元に変換する際に使用する。
        img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        # print(('img_tok',img_tok)) # tensor([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],...[0, 0, 0, 0, 0]]
        # print(('img_tok',img_tok.shape)) # torch.Size([32, 5])

        # 画像をresnet152で画像ベクトルに変換
        img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048
        # print(('img',img)) # tensor([[[1.8052, 0.4546, 2.8586,  ..., 0.0576, 0.3066, 0.3843],
        # print(('img',img.shape)) # torch.Size([32, 3, 2048])

        # 3*2048次元の画像ベクトルを5*768次元に変換。bertと同様にCLSとSEPベクトルを追加
        img_embed_out = self.img_embeddings(img, img_tok)
        # print(('img_embed_out',img_embed_out)) # tensor([[[ 0.1873, -0.3175, -0.3624,  ..., -0.0306,  0.0425,  0.1822],
        # print(('img_embed_out',img_embed_out.shape)) # torch.Size([32, 5, 768])

        # print(('input_txt',input_txt)) # tensor([[ 3608,  2100,  2322, 10376,  7530,  3122,  2184,  5898, 15357,  4524,
        # print(('input_txt',input_txt.shape)) # torch.Size([32, 24])
        save_pickle(input_txt, "./tmp/" + str(i_epoch) + "/" + "input_txt.pkl")
        # 単語ID化したテキストをbertで768次元に変換
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        # print(('txt_embed_out',txt_embed_out)) # tensor([[[-0.1270, -0.6901, -0.6514,  ..., -0.5027, -0.1357,  0.1038],
        # print(('txt_embed_out',txt_embed_out.shape)) # torch.Size([32, 24, 768])
        save_pickle(txt_embed_out, "./tmp/" + str(i_epoch) + "/" + "txt_embed_out.pkl")
        # 画像ベクトルとテキストベクトルをcat
        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID
        # print(('encoder_input',encoder_input)) # tensor([[[ 0.1873, -0.3175, -0.3624,  ..., -0.0306,  0.0425,  0.1822],
        # print(('encoder_input',encoder_input.shape)) # torch.Size([32, 29, 768])
        save_pickle(encoder_input, "./tmp/" + str(i_epoch) + "/" + "encoder_input.pkl")
        # catしたBertEncoder処理をする。encoded_layersは配列で返す。output_all_encoded_layers=Falseにすることで、最終1層のみ返ってくるようにする。
        ## https://github.com/Meelfy/pytorch_pretrained_BERT/blob/1a95f9e3e5ace781623b2a0eb20d758765e61145/pytorch_pretrained_bert/modeling.py#L239
        encoded_layers, attention_probs = self.encoder(
            encoder_input, extended_attention_mask, output_all_encoded_layers=False
        )

        # encoded_layersを保存する処理
        save_pickle(
            encoded_layers[-1], "./tmp/" + str(i_epoch) + "/" + "encoded_layers.pkl"
        )
        save_pickle(
            self.pooler(encoded_layers[-1]),
            "./tmp/" + str(i_epoch) + "/" + "self.pooler(encoded_layers[-1]).pkl",
        )
        # print(('encoded_layers',encoded_layers)) # [tensor([[[-1.2899e-01,  1.9738e-02,  4.5480e-01,  ..., -1.0199e-02],...]]],device='cuda:0', grad_fn=<AddBackward0>)
        # print(('encoded_layers',len(encoded_layers))) # 1
        # print(('encoded_layers[-1]',encoded_layers[-1])) # tensor([[[-1.0737e-01, -2.5804e-01,  3.6498e-01,  ...,  1.3026e-01,
        # print(('encoded_layers[-1]',encoded_layers[-1].shape)) # torch.Size([32, 29, 768]
        return (
            self.pooler(encoded_layers[-1]),
            attention_probs,
        )  # self.poolerdでclsベクトルを取ってくる


class MultimodalBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalBertClf, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, txt, mask, segment, img):
        x, attention_probs = self.enc(txt, mask, segment, img)
        return self.clf(x), attention_probs


def save_pickle(obj, path):
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)
