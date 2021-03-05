# Reference: https://github.com/guacomolia/ptr_net
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from utils import to_var

class Multi_spans_decoder(nn.Module):
    def __init__(self, args,  span_label_size = 2, is_GRU=True):
        super(Multi_spans_decoder, self).__init__()

        self.encoder_hidden_size = args.encoder_hidden_size
        self.decoder_hidden_size = args.decoder_hidden_size
        # self.answer_seq_len = answer_seq_len
        self.weight_size =args.attention_hidden_size
        self.is_GRU = is_GRU
        self.span_label_size = span_label_size

        # self.Wd = nn.Linear(2 * self.encoder_hidden_size, 2 * self.encoder_hidden_size, bias=False) # blending encoder
        self.Wh = nn.Linear(2 * self.encoder_hidden_size, self.decoder_hidden_size, bias=False) # blending encoder
        # self.Ws = nn.Linear(2 * self.encoder_hidden_size, self.decoder_hidden_size, bias=False) # blending encoder
    
        # self.emb = nn.Embedding(input_size, emb_size)  # embed inputs Embedding(11, 32)
        if is_GRU:
            # self.enc = nn.GRU(emb_size, hidden_size, batch_first=True)
            self.dec = nn.GRUCell(2 * self.encoder_hidden_size, self.decoder_hidden_size) # GRUCell's input is always batch first
        else:
            # self.enc = nn.LSTM(emb_size, hidden_size, batch_first=True)
            self.dec = nn.LSTMCell(2 * self.encoder_hidden_size, self.decoder_hidden_size) # LSTMCell's input is always batch first

        self.W1 = nn.Linear(2 * self.encoder_hidden_size, self.weight_size, bias=False) # blending encoder
        self.W2 = nn.Linear(self.decoder_hidden_size, self.weight_size, bias=False) # blending decoder
        self.start_weight = nn.Linear(self.weight_size, span_label_size, bias=False) # scaling sum of enc and dec by v.T
        self.end_weight = nn.Linear(self.weight_size, span_label_size, bias=False) # scaling sum of enc and dec by v.T


    def forward(self, input, state_out):
        """
        input:[batch, max_len, dim]
        state_out:[batch,dim]
        context_mask:[batch, max_len, dim]
        """
        batch_size = input.size(0)
        # Encoding
        # encoder_states, hc = self.enc(input) # encoder_state: (bs, L, H)
        encoder_states = input.transpose(1, 0) # (max_len, batch, dim)

        # Decoding states initialization
        decoder_input = state_out # (bs, embd_size)
        hidden = self.Wh(state_out)  # (bs, h)
        cell_state = state_out   # (bs, h) #

        probs = []
        max_len = encoder_states.size(0) #[max_len]
        outputs_s =  Variable(torch.zeros(max_len, batch_size, self.span_label_size)).cuda()
        outputs_e =  Variable(torch.zeros(max_len, batch_size, self.span_label_size)).cuda()

        # Decoding
        for i in range(max_len): # range(M)
            if self.is_GRU:
                hidden = self.dec(decoder_input, hidden) # (bs, h), (bs, h)
            else:
                hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (bs, h), (bs, h)
            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_states)          # (max_len, bs, weight_size)
            blend2 = self.W2(hidden)                  # (bs, wweight_size)
            blend_sum = F.tanh(blend1 + blend2)    # (max_len bs, Weight_size)
            start_logits = self.start_weight(blend_sum)     # (max_len, batch, 2)
            # print('out.size = ', out.size())
            #mask the output
            # start_logits = start_logits.transpose(0, 1) #[batch, max_len, 2]
            start_logits = F.softmax(start_logits, -1) #[max_len,batch, 2]
            outputs_s[i] = start_logits[i]

            end_logits = self.end_weight(blend_sum)     # (max_len, batch, 2)
            # print('out.size = ', out.size())
            #mask the output
            # end_logits = end_logits.transpose(0, 1) #[batch, amx_len, 2]
            end_logits = F.softmax(end_logits, -1) #[max_len,batch, 2]
            outputs_e[i] = end_logits[i]

        outputs_s = outputs_s.transpose(0,1) #[batch, max_len, span_label_size]
        outputs_e = outputs_e.transpose(0,1) #[batch, max_len, span_label_size]
        # (bs, M, L)
        return outputs_s, outputs_e

        