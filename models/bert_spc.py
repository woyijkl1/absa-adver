# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention
import ipdb


class BertPooler(nn.Module):
    def __init__(self):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.att = Attention(opt.bert_dim,score_function='mlp')
        self.pool = BertPooler()
        self.conv = nn.MaxPool1d(12)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids,text_bert_mask = inputs[0], inputs[1], inputs[2]
        
        # text_bert_len = torch.sum(text_bert_indices != 0, dim=-1)
        # text_bert_indices = self.squeeze_embedding(text_bert_indices, text_bert_len)
        # bert_segments_ids = self.squeeze_embedding(bert_segments_ids, text_bert_len)
        encoded_layers, pooled_output = self.bert(text_bert_indices, bert_segments_ids,text_bert_mask, output_all_encoded_layers=True)
        # cls_all_layers = torch.stack([i[:,0,:] for i in encoded_layers],dim=1)
        # cls_att, score = self.att(cls_all_layers,pooled_output)
        # cls_att = cls_att.squeeze(1)
        # ipdb.set_trace()
        # cls_all = torch.stack([i[:,-1,:] for i in encoded_layers],dim=1).permute(0,2,1)
        # cls_single = self.conv(cls_all).squeeze(-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits




