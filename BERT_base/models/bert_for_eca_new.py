import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers.modeling_bert import BertPreTrainedModel
from .transformers.modeling_bert import BertModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from .layers.GRU_decoder_new import Decoder
from .layers.pointer_network import PointerNetwork_Multi
from .layers.crf import CRF
from .layers.multi_spans_decoder import Multi_spans_decoder
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from tools.finetuning_argparse_eca import get_argparse 
from torch.autograd import Variable
from processors_eca.eca_seq import bert_extract_item

args = get_argparse().parse_args()   

class Bert2Crf(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert2Crf, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.label_size)
        self.crf = CRF(num_tags=args.label_size, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,context_mask = None, labels=None, span_labels = None, start_positions=None,end_positions=None, testing = False):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output = torch.mul(sequence_output, context_mask.unsqueeze(-1).repeat(1,1,sequence_output.size(-1))) #[batch, max_len, -1, dim] 将cls去掉
        logits = self.classifier(sequence_output)#[batch, max_len, label_size]
        outputs = (logits,)
        if labels is not None:
            # print('logits = ', logits.size())
            # print('labels = ', labels.size())
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores


class Bert2Gru(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert2Gru, self).__init__(config)
        self.bert = BertModel(config)
   
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.decoder = Decoder(args, num_classes=args.label_size, dropout=0.2)
        self.clsdense = nn.Linear(config.hidden_size, args.decoder_hidden_size)
       
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, context_mask = None, labels=None, span_labels = None, start_positions=None,end_positions=None, testing = False):
        #注意这里的target是根据长度排序过的数据
        x_mask = attention_mask > 0 #score: 原始数据，是词语的index [8,83] source_mask:tru false的数组 组成的【true， false】
        x_len = torch.sum(x_mask.int(), -1)
   
        target_= labels # 是一个PackedSequence的数据
        max_len = input_ids.size(1) #in other sq2seq, max_len should be target.size() batch——size
        batch_size = input_ids.size(0) 
        if labels != None:
            # print('rrr = ', labels)
            target, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, total_length=max_len) #target [83, 8]

        #对x进行编码
        outputs_x =self.bert(input_ids = input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids)
        encoder_outputs = outputs_x[0]
        cls_out = encoder_outputs[:,0,:] #[batch, dim],第0个向量CLS
        #mask掉情绪表达的部分
        encoder_outputs = torch.mul(context_mask.unsqueeze(-1).repeat(1,1,encoder_outputs.size(-1)), encoder_outputs)
        encoder_outputs = self.dropout(encoder_outputs) #[batch, max_len, dim]

        hidden =  cls_out.unsqueeze(0).repeat(args.decoder_num_layers, 1, 1)# [args.decoder_num_layers, batch, hiddendim]
        hidden = self.clsdense(hidden)

        label_size = args.label_size
        outputs =  Variable(torch.zeros(max_len, batch_size, label_size)).cuda()
        output = Variable(torch.zeros((batch_size))).long().cuda()
        encoder_outputs = encoder_outputs.transpose(0,1) #[max_len, batch, dim]
        for t in range(max_len):
            current_encoder_outputs = encoder_outputs[t,:,:].unsqueeze(0) #[1, batch, 2*encoder_hidden_size] 第t个词语的表示，去掉第一个维度，【batch, 2*dim】
            output, hidden = self.decoder(output, hidden, current_encoder_outputs)
            outputs[t] = output
            #is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if testing:
                output = Variable(top1).cuda()
            else:
                output = Variable(target[t]).cuda()
        
        pre_lebels = outputs.transpose(0,1).argmax(-1) #[batch, max_len]
        if testing:
            # outputs = outputs.transpose(0,1)
            return pre_lebels
        else:
            # print('x_len = ', x_len)
            # print('rr =', outputs)
            packed_y = torch.nn.utils.rnn.pack_padded_sequence(outputs, x_len.cpu())
            loss  = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y.data), target_.data)
            return (loss,)


class Bert_softmax(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_softmax, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.label_size)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, context_mask = None, span_labels = None, start_positions=None,end_positions=None, labels=None, testing = False):
        """
        input_ids:[batch, max_len]
        token_type_ids:
        attention_mask:
        emotion_con_mask: #可通过token_type_ids获得
        tagging的计算
        """
        outputs =self.bert(input_ids = input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
      
        sequence_output = torch.mul(sequence_output, context_mask.unsqueeze(-1).repeat(1,1,sequence_output.size(-1))) #[batch, max_len, dim] 将cls去掉
        x_len = torch.sum(attention_mask.int(), -1)  #[batch]
        logits = self.classifier(sequence_output) #[batch, max_len, 3]
        pre_labels = logits.argmax(-1) #[batch, max_len]
        if testing:
            return pre_labels
        else:
            logits = logits.transpose(0,1) #[max_len, bacth, dim]
            packed_y = torch.nn.utils.rnn.pack_padded_sequence(logits, x_len.cpu())
            loss  = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y.data), labels.data)
            return (loss, pre_labels)


class Bert_Multi_Point(BertPreTrainedModel):
    """
   
    """
    def __init__(self, config):
        super(Bert_Multi_Point, self).__init__(config)
        self.bert = BertModel(config)
        self.pointer = PointerNetwork_Multi(args, answer_seq_len = args.answer_seq_len, is_GRU=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, context_mask = None, span_label = None, start_positions=None,end_positions=None, labels=None, testing = False):
        """
        input_ids:[batch, max_len]
        token_type_ids:
        attention_mask:
        emotion_con_mask: #可通过token_type_ids获得
        span_labels：[batch, max_len, max_span_len]
        """
        outputs =self.bert(input_ids = input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        # print('size = ', sequence_output.size())
        cls_out = sequence_output[:,0,:] #[batch, dim],第0个向量CLS
        sequence_output = self.dropout(sequence_output)
        # sequence_output = sequence_output.transpose(0,1) #[batch, max_len, dim]
        sequence_output = torch.mul(sequence_output, attention_mask.unsqueeze(-1).repeat(1,1,sequence_output.size(-1))) #[batch, max_len, dim] 将cls去掉
        
        probs, logits = self.pointer(sequence_output, cls_out, attention_mask)#[batch, max_len], [batch, max_len]

        if testing:
            start_logits = logits[0] #[batch, ans_len, max_len]
            end_logits = logits[1] #[batch, ans_len, max_len]
            start_label = torch.argmax(start_logits, -1) #[batch, ans_len]
            end_label = torch.argmax(end_logits, -1) #[batch, ans_len]
            return (start_label, end_label) 
        else:
            max_len = sequence_output.size(1)
            outputs = probs.view(-1, max_len) # (bs*M, L)
            # print('y = ', y.size())
            span_label = span_label.reshape(-1)# (bs*M)
            loss = F.nll_loss(outputs, span_label)
            return (loss, logits)

