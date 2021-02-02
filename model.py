import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear


class Model(nn.Module):

    def __init__(self, vocab_size, word_emb_init, hid_dim, num_classes):
        super(Model, self).__init__()
        """
        vocab_size(int):单词量
        word_emb_init(str):word embedding预训练权重路径
        hid_dim(int):用与GRU->512
        num_classes(int):输出类别->3129
        """
        emb_dim = int(word_emb_init.split('_')[-1].split('.')[0])
        self.hid_dim = hid_dim

        '''
        问题的encoding编码
        '''
        # 对vocab_size个词汇进行长度为emb_dim的Embedding，生成词向量
        self.word_embed = nn.Embedding(vocab_size + 1, emb_dim)
        # 输入特征为emb_dim大小，隐含层大小为hid_dim的GRU，输入词向量训练
        self.gru = nn.GRU(emb_dim, hid_dim)

        '''
        图像的attention注意力
        '''
        # [batch_size, hid_dim] => [hid_dim, 1]
        # 输出为长度为一的标量: attention权重
        self.att = nn.Linear(hid_dim, 1)

        '''
        输出classifier分类器
        '''
        # [batch_size, hid_dim] => [hid_dim, num_classes]
        self.clf = nn.Linear(hid_dim, num_classes)
        # 进行0.5的Dropout
        self.clf_do = nn.Dropout(0.5, inplace=True)

        '''
        初始化单词Embedding层权重
        '''
        # 初始化词向量权重为[vocab_size+1, emb_dim]大小的float32零矩阵
        pretrained_word_embed = np.zeros((vocab_size + 1, emb_dim), dtype=np.float32)
        # 将初始化词向量权重word_emb_init赋值到预训练词向量权重中pretrained_word_embed
        pretrained_word_embed[:vocab_size] = np.load(word_emb_init)
        # 将Embedding层word_embed的权重初始化
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_word_embed))

        '''
        门控tanh激活器
        '''
        self.gth_iatt = nn.Linear(2048 + hid_dim, hid_dim)
        self.gthp_iatt = nn.Linear(2048 + hid_dim, hid_dim)
        self.gth_q = nn.Linear(hid_dim, hid_dim)
        self.gthp_q = nn.Linear(hid_dim, hid_dim)
        self.gth_i = nn.Linear(2048, hid_dim)
        self.gthp_i = nn.Linear(2048, hid_dim)
        self.gth_clf = nn.Linear(hid_dim, hid_dim)
        self.gthp_clf = nn.Linear(hid_dim, hid_dim)

    # end to end !!   :)
    def forward(self, image, question):
        """
        question -> shape(batch, 14)
        image -> shape(batch, 36, 2048)
        """

        '''
        问题的encoding编码
        '''
        emb = self.word_embed(question)            # (batch, seqlen, emb_dim)
        enc, hid = self.gru(emb.permute(1, 0, 2))  # (seqlen, batch, hid_dim)
        qenc = enc[-1]                             # (batch, hid_dim)

        '''
        图像的attention注意力
        '''
        qenc_reshape = qenc.repeat(1, 36).view(-1, 36, self.hid_dim)  # (batch, 36, hid_dim)
        image = F.normalize(image, -1)                                # (batch, 36, 2048)
        concated = torch.cat((image, qenc_reshape), -1)               # (batch, 36, 2048 + hid_dim)
        concated = torch.mul(torch.tanh(self.gth_iatt(concated)), torch.sigmoid(self.gthp_iatt(concated)))
        a = self.att(concated)                               # (batch, 36, 1)
        a = F.softmax(a.squeeze(), dim=1)                    # (batch, 36)
        v_head = torch.bmm(a.unsqueeze(1), image).squeeze()  # (batch, 2048)

        # （问题 + 图像）
        q = torch.mul(torch.tanh(self.gth_q(qenc)), torch.sigmoid(self.gthp_q(qenc)))
        v = torch.mul(torch.tanh(self.gth_i(v_head)), torch.sigmoid(self.gthp_i(v_head)))
        h = torch.mul(q, v)  # (batch, hid_dim)

        '''
        输出classifier分类器
        '''
        s_head = self.clf(torch.mul(torch.tanh(self.gth_clf(h)), torch.sigmoid(self.gthp_clf(h))))
        s_head = self.clf_do(s_head)

        return s_head
