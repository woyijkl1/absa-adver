# -*- coding: utf-8 -*-
# file: atae-lstm
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.squeeze_embedding import SqueezeEmbedding


class ATAE_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

        self.cls_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.cls_asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.h_h_proj = nn.Linear(4*opt.hidden_dim,opt.hidden_dim)
        self.cls_dense = nn.Linear(opt.hidden_dim,opt.polarities_dim)
        self.dense_total = nn.Linear(opt.hidden_dim+opt.hidden_dim, opt.polarities_dim)

        self.conv = nn.Conv1d(300,2*opt.hidden_dim,3)


    def classifier(self,inputs):
        text_raw_indices = inputs[0] # batch_size x seq_len
        aspect_indices = inputs[1] 
        batch_size = text_raw_indices.size(0)
        ctx_len = torch.sum(text_raw_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)


        ctx = self.embed(text_raw_indices) # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices) # batch_size x seq_len x embed_dim

        # use cnn
        ctx = ctx.permute(0,2,1)
        asp = asp.permute(0,2,1)
        cnn_ctx = self.conv(ctx)
        cnn_asp = self.conv(asp)
        cnn_cxt = F.max_pool1d(cnn_ctx,cnn_ctx.size(2)).squeeze(-1)
        cnn_asp = F.max_pool1d(cnn_asp,cnn_asp.size(2)).squeeze(-1)
        context_h = cnn_cxt
        asp_h = cnn_asp


        # cls_out, (context_h, _) = self.cls_lstm(ctx, ctx_len)
        # asp_out, (asp_h, _) = self.cls_asp_lstm(asp, asp_len)
        # context_h = context_h.permute(1,0,2).view(batch_size,-1)
        # asp_h = asp_h.permute(1,0,2).view(batch_size,-1)

        h_h = torch.cat([context_h,asp_h],dim=-1)
        h_h = F.tanh(self.h_h_proj(h_h))

        # cls_asp = torch.cat((hn,asp_mean),dim=-1)
        out = self.cls_dense(h_h)
        return out,h_h



    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float).to(self.opt.device)

        x = self.embed(text_raw_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)

        out = self.dense(output)

        # for classifier
        if self.opt.classifier:
            class_type,cls_asp = self.classifier(inputs)
            cls_feat = torch.cat((cls_asp,output),dim=-1)
            out = self.dense_total(cls_feat) # bathc_size x polarity_dim
            return out



        return out
