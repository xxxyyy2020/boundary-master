# Reference: https://github.com/guacomolia/ptr_net
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PointerNetwork_Multi(nn.Module):
    def __init__(self, args, answer_seq_len = 2, is_GRU=True):
        super(PointerNetwork_Multi, self).__init__()
        self.encoder_hidden_size = args.encoder_hidden_size
        self.decoder_hidden_size = args.decoder_hidden_size
        self.answer_seq_len = answer_seq_len
        self.weight_size =args.attention_hidden_size
        self.is_GRU = is_GRU

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


        self.vts = torch.nn.ModuleList([nn.Linear(self.weight_size, 1, bias=False) for i in range(self.answer_seq_len)])
        # self.vt = nn.Linear(self.weight_size, 1, bias=False) # scaling sum of enc and dec by v.T


    def forward(self, input, state_out, context_mask):
        """
        input:[batch, max_len, dim]
        context_mask:[batch, max_len, dim]
        """
        batch_size = input.size(0)
        # Encoding
        # encoder_states, hc = self.enc(input) # encoder_state: (bs, L, H)
        encoder_states = input.transpose(1, 0) # (max_len, batch, dim)

        # Decoding states initialization
        decoder_input = state_out # (bs, embd_size)
        hidden = self.Wh(state_out)  # (bs, h)
        cell_state = state_out  

        max_len = input.size(1)
        probs_s = Variable(torch.zeros((self.answer_seq_len, batch_size, max_len))).long().cuda()
        probs_e = Variable(torch.zeros((self.answer_seq_len, batch_size, max_len))).long().cuda()

        probs = []
        # Decoding
        index_fun = 0
        for i in range(self.answer_seq_len * 2): # range(M)
            if i != 0 and i % 2 == 0:
                index_fun += 1
            
            if self.is_GRU:
                hidden = self.dec(decoder_input, hidden) # (bs, h), (bs, h)
            else:
                hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (bs, h), (bs, h)
            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_states)          # (max_len, bs, weigh_size)
            blend2 = self.W2(hidden)                  # (batch, weih_size)
            blend_sum = F.tanh(blend1 + blend2)    # (max_len, bs, Wweig_size)

            func1 = self.vts[index_fun]
            out = func1(blend_sum).squeeze(-1)        # (max_len, bs)
            # print('out.size = ', out.size())
            #mask the output
            out = out.transpose(0, 1) #[batch, max_len]
            out = out.masked_fill(context_mask == 0, -1e9)
            out = F.log_softmax(out, -1) #[batch, max_len]
            # out = F.log_softmax(out.contiguous(), -1) # (bs, L)
            if i % 2 == 0:
                probs_s[index_fun] = out
            else:
                probs_e[index_fun] = out
            probs.append(out)

        probs = torch.stack(probs, dim=1)#[ans * 2 * batch, max_len]
        # (bs, M, L)
        return probs, (probs_s.transpose(0,1), probs_e.transpose(0,1)) #[ans_lan*2*batch, max_len], [batch, ans_len, max_len], [batch, ans_len, max_len]


