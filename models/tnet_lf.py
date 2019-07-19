import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy
from layers.dynamic_rnn import DynamicLSTM
import ipdb


class Absolute_Position_Embedding(nn.Module):
    def __init__(self, opt, size=None, mode='sum'):
        self.opt = opt
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Absolute_Position_Embedding, self).__init__()

    def forward(self, x, pos_inx):
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.size(-1))
        batch_size, seq_len = x.size()[0], x.size()[1]
        weight = self.weight_matrix(pos_inx, batch_size, seq_len).to(self.opt.device)
        x = weight.unsqueeze(2) * x
        return x



    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][1]):
                relative_pos = pos_inx[i][1] - j
                weight[i].append(1 - relative_pos / 40)
            for j in range(pos_inx[i][1], seq_len):
                relative_pos = j - pos_inx[i][0]
                weight[i].append(1 - relative_pos / 40)
        weight = torch.tensor(weight)
        return weight

class TNet_LF(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TNet_LF, self).__init__()
        print("this is TNet_LF model")
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.position = Absolute_Position_Embedding(opt)
        self.opt = opt
        D = opt.embed_dim  # 模型词向量维度
        C = opt.polarities_dim  # 分类数目
        L = opt.max_seq_len
        HD = opt.hidden_dim
        self.lstm1 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.convs3 = nn.Conv1d(2 * HD, 50, 3, padding=1)
        self.fc1 = nn.Linear(4 * HD, 2 * HD)
        self.fc = nn.Linear(50, C)

        self.dense_total = nn.Linear(70, opt.polarities_dim)
        self.h_h_proj = nn.Linear(4*opt.hidden_dim,20)
        self.cls_dense = nn.Linear(20,3)
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
        text_raw_indices, aspect_indices, aspect_in_text = inputs[0], inputs[1], inputs[2]
        feature_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        feature = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        v, (_, _) = self.lstm1(feature, feature_len)
        e, (_, _) = self.lstm2(aspect, aspect_len)
        v = v.transpose(1, 2)
        e = e.transpose(1, 2)
        for i in range(2):
            a = torch.bmm(e.transpose(1, 2), v)
            a = F.softmax(a, 1)  # (aspect_len,context_len)
            aspect_mid = torch.bmm(e, a)
            aspect_mid = torch.cat((aspect_mid, v), dim=1).transpose(1, 2)
            aspect_mid = F.relu(self.fc1(aspect_mid).transpose(1, 2))
            v = aspect_mid + v
            v = self.position(v.transpose(1, 2), aspect_in_text).transpose(1, 2)
        z = F.relu(self.convs3(v))  # [(N,Co,L), ...]*len(Ks)
        z = F.max_pool1d(z, z.size(2)).squeeze(2)
        

         # for classifier
        if self.opt.classifier:
            class_type,cls_asp = self.classifier(inputs)
            cls_feat = torch.cat((cls_asp,z),dim=-1)
            out = self.dense_total(cls_feat) # bathc_size x polarity_dim
            return out
        else:
            out = self.fc(z)
            return out




        return out
