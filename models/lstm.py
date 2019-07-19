# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

        
        self.cls_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.cls_asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.h_h_proj = nn.Linear(4*opt.hidden_dim,opt.hidden_dim)
        self.cls_dense = nn.Linear(opt.hidden_dim,opt.polarities_dim)
        self.dense_total = nn.Linear(opt.hidden_dim+opt.hidden_dim, opt.polarities_dim)
        self.conv = nn.Conv1d(300,2*opt.hidden_dim,3)

        self.conv_gating = nn.Conv1d(300,2*opt.hidden_dim,3)
        self.dense_gating = nn.Linear(2*opt.hidden_dim,opt.polarities_dim_gating)
        


    def gating(self,inputs):
        text_raw_indices = inputs[0]
        ctx_len = torch.sum(text_raw_indices != 0, dim=1)
        ctx = self.embed(text_raw_indices) 
        ctx = ctx.permute(0,2,1)
        cnn_ctx = self.conv_gating(ctx)
        cnn_ctx = F.max_pool1d(cnn_ctx,cnn_ctx.size(2)).squeeze(-1)
        out = self.dense_gating(cnn_ctx)
        return out, cnn_ctx







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
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])

        # for classifier
        if self.opt.classifier:
            class_type,cls_asp = self.classifier(inputs)
            cls_feat = torch.cat((cls_asp,h_n[0]),dim=-1)
            out = self.dense_total(cls_feat) # bathc_size x polarity_dim

            return out



        return out
