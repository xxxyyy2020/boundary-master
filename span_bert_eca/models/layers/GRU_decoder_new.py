#encoding=utf-8
import argparse
import torch
import time
import json
import numpy as np
import math
import random
from torch.autograd import Variable
import torch.nn.functional as F

class Decoder(torch.nn.Module):
    def __init__(self, args, num_classes=3, dropout=0.2):
        super(Decoder, self).__init__()
        self.args = args
        self.label_embedding = torch.nn.Embedding(num_classes, args.label_embedding_size)
        self.dropout = torch.nn.Dropout(args.dropout)
        # self.attention = Attention(args.encoder_hidden_size, args.decoder_hidden_size, args.attention_hidden_size)
        self.rnn = torch.nn.GRU(args.encoder_hidden_size*2 + args.label_embedding_size, args.decoder_hidden_size, args.decoder_num_layers, batch_first=False, bidirectional=False)
        self.hidden2label = torch.nn.Linear(args.decoder_hidden_size, num_classes)

    def forward(self, label, last_hidden, current_encoder_outputs):
        """
        inputs: [batch],
        last_hiddeen: [layer, batch, hidden]
        encoder_outputs:[max_len, batch, 2*hidden]
        current_encoder_outputs: [1, batch, 2*hidden]
        time_step:代表解码第time_step个词语
        max_len：句子的最大长度
        """
        embedded = self.label_embedding(label).unsqueeze(0) #[1, batch, label_embedding_size]
        embedded = self.dropout(embedded)

        rnn_inputs = torch.cat([embedded, current_encoder_outputs], 2) #[batch, 2*hidden+lable_size]
        output, hidden = self.rnn(rnn_inputs, last_hidden)
        output = output.squeeze(0)
        
        output = self.hidden2label(output)
        output = F.log_softmax(output, dim=1)

        return output, hidden



        