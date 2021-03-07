"""
1.获取数据属性
2.记录数据
"""
from transformers import BertTokenizer
import torch
import copy
import pickle
import numpy as np
import re

'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    content = pickle.load(pkl_file)
    pkl_file.close()
    return content

def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()


def remove_noise_strr(content):
    """
    args:
        :param content:
        :return:
    """
    content = content.replace('@', ' ')
    content = content.replace('#', ' ')
    content = content.replace('<', ' ')
    content = content.replace('>', ' ')
    content = content.replace('=', ' ')
    # content = content.replace('`', ' ')
    content = content.replace('�', ' ')
    content = content.replace('\n', '')
    content = content.replace('\t', ' ')
    content = content.replace('*', ' ')
    content = content.replace('&', '')
    content = content.replace('&', '')
   
    content = content.replace('  ', ' ')
    content = content.replace('  ', ' ')
    content = content.replace('  ', ' ')
    return content.strip()


def batch_generator(x_ids, x_seg_ids, x_input_masks,target_ids, examples, batch_size=128, return_idx=False):
    # print(x.shape)
    # print(y.shape[0])
    # print(len(examples))
    
    for offset in range(0, x_ids.shape[0], batch_size):
    # for offset in range(0, 2*batch_size, batch_size):
        batch_example = examples[offset:offset+batch_size]
        x_input_masks_batch = x_input_masks[offset:offset+batch_size]
        batch_x_len = np.sum(x_input_masks_batch, -1)
        # print('batch_x_len = ', batch_x_len)
        # batch_x_len = [sum(example['data_len']) for example in batch_example]
        # batch_x_len = np.array(batch_x_len)
        # batch_idx=batch_x_len.argsort()[::-1]
        batch_idx= batch_x_len.argsort()[::-1]
        max_doc_len =  max(batch_x_len) #文本的最大长度
        # print('batch_idx = ', batch_idx)

        # batch_xe_len= [batch_example[i]['data_emo_len'] for i in batch_idx] #因为情感词语的长度
        # batch_xe_len= np.array(batch_xe_len)
        # batch_xe_len = np.sum(x_e_input_masks[offset:offset+batch_size], -1)
        # print('max_doc_len = ', max_doc_len)
        # batch_x_len=batch_x_len[batch_idx]
        # print('x_ids = ',x_ids)
        new_end = offset+batch_size
        batch_x = x_ids[offset:new_end][batch_idx] 
        batch_seg_x = x_seg_ids[offset:new_end][batch_idx] 
        batch_input_masks_x = x_input_masks[offset:new_end][batch_idx] 
        batch_x_len=batch_x_len[batch_idx]
        # batch_xe_len = batch_xe_len[batch_idx]

        # batch_xe = x_e_ids[offset:new_end][batch_idx] 
        # batch_seg_xe = x_e_seg_ids[offset:new_end][batch_idx] 
        # batch_input_masks_xe = x_e_input_masks[offset:new_end][batch_idx] 

        batch_y = target_ids[offset:new_end][batch_idx]
        raw_y = target_ids[offset:new_end][batch_idx]
        

        batch_x_             = batch_x[:, :max_doc_len]
        batch_seg_x_         = batch_seg_x[:, :max_doc_len]
        batch_input_masks_x_ = batch_input_masks_x[:, :max_doc_len]

        # batch_xe_             = batch_xe[:, :max_doc_len]
        # batch_seg_xe_         = batch_seg_xe[:, :max_doc_len]
        # batch_input_masks_xe_ = batch_input_masks_xe[:, :max_doc_len]

        batch_y  = batch_y[:, :max_doc_len]
        raw_y    = raw_y[:, :max_doc_len]

        #转换为torch类型
        # print('batch_x = ',batch_x)
        batch_x_             = torch.from_numpy(np.array(batch_x_)).long().cuda()
        batch_seg_x_         = torch.from_numpy(np.array(batch_seg_x_)).long().cuda()
        batch_input_masks_x_ = torch.from_numpy(np.array(batch_input_masks_x_)).long().cuda()

        # batch_xe_             = torch.from_numpy(np.array(batch_xe_)).long().cuda()
        # batch_seg_xe_        = torch.from_numpy(np.array(batch_seg_xe_)).long().cuda()
        # batch_input_masks_xe_ = torch.from_numpy(np.array(batch_input_masks_xe_)).long().cuda()

        batch_y  = torch.from_numpy(np.array(batch_y)).long().cuda()
        batch_x_len_ = torch.from_numpy(np.array(batch_x_len)).long().cuda()
        # batch_xe_len_ = torch.from_numpy(np.array(batch_xe_len)).long().cuda()

        # print('batch_idx = ', batch_idx)
        # print('len(batch_example ) = ', len(batch_example))

        batch_exams = [batch_example[i] for i in batch_idx]
        # assert x.shape[0] == y.shape[0] == len(examples) == x_e.shape[0]
        
        if len(batch_y.size() )==2:
            batch_y=torch.nn.utils.rnn.pack_padded_sequence(batch_y, batch_x_len, batch_first=True)
        # if return_idx: #in testing, need to sort back.
            
        #     yield (batch_x, batch_seg_x, batch_input_masks_x, batch_xe, batch_seg_xe, batch_input_masks_xe, batch_x_len, batch_xe_len, batch_y, batch_exams, raw_y)
        # else:
        yield (batch_x_, batch_seg_x_, batch_input_masks_x_, batch_x_len_, batch_y, batch_exams, raw_y)



def check(cause, current_clause):
    """ 
    cause
    current_clause
    """
    # cause_l = cause.lower().split()
    # print('current_clause = ', current_clause)
    # print('cause = ', cause)

    cause_str = ''
    for index, item in enumerate(cause):
        item = item.replace('#', '')
        cause_str += item
    cause_str.strip()
    
    clause_str = ''
    for index, item in enumerate(current_clause):
        clause_str += item.replace('#', '')
    clause_str.strip()

    # print('clause_str = ', clause_str)
    # print('cause_str = ', cause_str)

    span = re.search(re.escape(cause_str), clause_str).span()
    start_index = span[0]
    end_index = span[1]
    # print('start_index = ', start_index)
    # print('end_index = ', end_index)

    str_len = 0
    tagg = 0
    start = -1
    end = -1
    for index, item in enumerate(current_clause):
        item = item.replace('#', '')
        if start_index == tagg:
            start = index
        if end_index == tagg + len(item):
            end = index + 1
     
        
        tagg = tagg + len(item)
        # print('tag =', tagg)

    if start_index == -1 or end_index == -1:
        # print('current_clause = ', current_clause)
        # print('cause = ', cause)
        raise Exception("Invalid level!", level)
    # print('start = ', start)
    # print('end = ', end)
    return start, end


punstr = [',','.', '"', '\'', ';', ':', '!', '?', '.']
def read_en_pkl(data_path):
    """
    将数据读取和写入
     [{'docId': 0}, 
     {'name': 'surprise', 'value': '5'}, 
     [{'keyword': 'was startled by', 'keyloc': 1, 'clauseID': 2}], 
     [{'index': 1, 'cause_content': 'his unkempt hair and attire', 'clauseID': 2}], 
     [{'cause': 'N', 'id': '1', 'keywords': 'N', 'clauseID': 1, 'content': 'That day Jobs walked into the lobby of the video game manufacturer Atari and told the personnel director'}, 
     {'cause': 'Y', 'id': '2', 'keywords': 'Y', 'clauseID': 2, 'content': 'who was startled by his unkempt hair and attire', 'cause_content': 'his unkempt hair and attire', 'key_content': 'was startled by'}, 
     {'cause': 'N', 'id': '3', 'keywords': 'N', 'clauseID': 3, 'content': "that he wouldn't leave until they gave him a job."}]]
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # data_path = os.path.join(args.raw_data_dir, "eca_en.pkl")
    data = loadList(data_path)
    # outputFile1 = codecs.open(save_txt_path, 'w','utf-8') #将文本写入到csv文件
    out_data= [] #是一个list列表，每一个list都存入一个字典
    for index, item in enumerate(data): #对于每一个文本
    #     data_dict['id'] =  str(len(all_data))
    #   data_dict['emotion'] = strr
    #   data_dict['text'] = text
    #   data_dict['cause'] = cause
        para_data = dict() #将每一个文本写成一个字典的形式
        # print('item = ', item)
        content_Data = remove_noise_strr(item['text']).strip()#每一个文本的内容
        docID = item['id'] #文本的ID号
        cause = remove_noise_strr(item['cause']).strip() #所有的子句信息
        emo_info = item['emotion']

        doc_token = tokenizer.tokenize(content_Data)
        cause_token = tokenizer.tokenize(cause)
        
        data_len = len(doc_token)
        start, end = check(cause_token, doc_token)

        target_l = ['O'] * len(doc_token)
        target_l[start] = 'B'
        for indd in range(start + 1, end):
            target_l[indd] = 'I'
           
        assert len(target_l) == len(doc_token)
       
        para_data['docID'] = docID
        para_data['emo_info'] = emo_info
        para_data['cause_token'] = cause_token
        para_data['doc_token'] = doc_token
        para_data['target_data'] = target_l
        para_data['data_len'] = data_len
        # para_data['data_emo_len'] = len(emotion_token)
        out_data.append(para_data)

    return out_data


def pad_data(data_example, max_len_doc):
    """
    data_example:
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    target_dic = ['O', 'B', 'I']
    # x_id, x_e_id, target_id = [], [], []
    x_ids, x_seg_ids, x_input_masks = [], [],[]
    # x_e_ids, x_e_seg_ids, x_e_input_masks = [], [],[]
    x_lens = []
    target_ids = []

    for index, item in enumerate(data_example):

        x_token = item['doc_token']
        x_clause_len = item['data_len']

        # x_emo_token = item['emotion_token']
        target_lable = item['target_data']

        if len(x_token) > max_len_doc - 2:
            x_token = x_token[0:max_len_doc - 2]
            target_lable = target_lable[0:max_len_doc - 2]
        # if len(x_emo_token) > max_len_doc - 2:
        #     x_emo_token = x_emo_token[0:max_len_doc - 2]
           
        #对x和xe和lable 都填充 [CLS],[SEP]
        x_token.insert(0, '[CLS]')
        x_token.append('[SEP]')
        # x_emo_token.insert(0, '[CLS]')
        # x_emo_token.append('[SEP]')
        target_lable.insert(0,'O')
        target_lable.append('O')

        #将最开始的子句以及后面的子句分别加1
        # print('x_clause_len = ', x_clause_len)
        # a = int(x_clause_len[0])
        # x_clause_len[0] = a + 1
        # b = int(x_clause_len[-1])
        # x_clause_len[-1] = b + 1
        x_len = len(x_token)
        # x_emo_len = len(x_emo_token)

        x_lens.append(x_len)
        # x_emo_lens.append(x_emo_len)

        #获取id信息
        # print('x_token =', x_token)
        x_id = tokenizer.convert_tokens_to_ids(x_token)
        # x_emo_id = tokenizer.convert_tokens_to_ids(x_emo_token)
        target_id = [target_dic.index(item) for item in target_lable]

        x_seg_id = [0] * len(x_id)
        x_input_mask = [1] * len(x_id)
        # x_emo_seg_id = [0] * len(x_emo_id)
        # x_emo_input_mask = [1] * len(x_emo_id)

        x_ids.append(x_id)
        x_seg_ids.append(x_seg_id)
        x_input_masks.append(x_input_mask)
        # x_e_ids.append(x_emo_id)
        # x_e_seg_ids.append(x_emo_seg_id)
        # x_e_input_masks.append(x_emo_input_mask)
        target_ids.append(target_id)

    #进行填充
    max_x_len = max(x_lens)
    # max_x_e_len = max(x_emo_lens)
    x_pad_ids, x_pad_seg_ids, x_pad_input_masks,  target_pad_ids = [],[],[],[]

    for index in range(0, len(x_lens)):#对每一个实体进行填充

        x_id = x_ids[index]
        x_seg_id = x_seg_ids[index]
        x_input_mask = x_input_masks[index]
        target_id = target_ids[index]

        while len(x_id) < max_x_len: # Zero-pad up to the sequence length.
            x_id.append(0)
            x_seg_id.append(0)
            x_input_mask.append(0)
            target_id.append(0)

        x_pad_ids.append(x_id)
        x_pad_seg_ids.append(x_seg_id)
        x_pad_input_masks.append(x_input_mask)
        target_pad_ids.append(target_id)
            
        assert len(x_id) == max_x_len
        assert len(x_seg_id) == max_x_len
        assert len(x_input_mask) == max_x_len
        assert len(target_id) == max_x_len

    print(np.array(x_pad_ids).shape, np.array(x_pad_seg_ids).shape, np.array(x_pad_input_masks).shape,  np.array(target_pad_ids).shape)
    assert len(x_pad_ids) == len(x_pad_seg_ids) == len(x_pad_input_masks) == len(target_pad_ids)
    
    return np.array(x_pad_ids), np.array(x_pad_seg_ids), np.array(x_pad_input_masks), np.array(target_pad_ids)
    # return x_pad_ids, x_pad_seg_ids, x_pad_input_masks, x_pad_e_ids, x_pad_e_seg_ids, x_pad_e_input_masks, target_pad_ids

