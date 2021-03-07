import csv
import json
import torch
from models.transformers import BertTokenizer
import pickle
import codecs
import re
import string
from processors_eca.func import loadList, saveList, get_clean_data_ch, get_clean_data_en

class EcaTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=True):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_en_pkl(self, data_path, save_csv_path = None):
        #获取每个数据的属性
        """
        要获取文本数据， 情感数据， 原因的位置，以及要进行核对原因位置是否正确
        获取example的list列表
        """
        """
        s = "string. With. Punctuation?"
        table = str.maketrans({key: ' ' + key + ' ' for key in string.punctuation})
        print('table = ', table)
        new_s = s.translate(table)
        [
        {'docId': 0}, 
        {'name': 'surprise', 'value': '5'}, 
        [{'keyword': 'was startled by', 'keyloc': 1, 'clauseID': 2}], 
        [{'index': 1, 'cause_content': 'his unkempt hair and attire', 'clauseID': 2}], 
        [{'cause': 'N', 'id': '1', 'keywords': 'N', 'clauseID': 1, 'content': 'That day Jobs walked into the lobby of the video game manufacturer Atari and told the personnel director'}, 
        {'cause': 'Y', 'id': '2', 'keywords': 'Y', 'clauseID': 2, 'content': 'who was startled by his unkempt hair and attire', 'cause_content': 'his unkempt hair and attire', 'key_content': 'was startled by'}, {'cause': 'N', 'id': '3', 'keywords': 'N', 'clauseID': 3, 'content': "that he wouldn't leave until they gave him a job."}]]
        """
        return_data = []
        data = loadList(data_path)

        for index, item in enumerate(data):

            example_dic =dict()
            docID = item[0]['docId']
            emotion_loc = int(item[2][0]['keyloc'])
            clause_info = item[4] #clause 信息

            emo_clause = clause_info[emotion_loc]['content']
            emotion_content = get_clean_data_en(emo_clause).split()

            example_dic['docID'] = docID
            example_dic['emo_data'] = emotion_content
            
            content_data = []
            target_data = []
            clause_len = []
            span_index = []

            tagg = 0

            for indexc, itemc in enumerate(clause_info):
                content_text =get_clean_data_en(itemc['content'])
                content_l = content_text.split()
               
                content_data.extend(content_l) #添加子句的word
                clause_len.append(len(content_l))#获取子句的长度
                #获取类别标签
                taget_l = ['O'] * len(content_l)
                ifcause = itemc['cause']

                if ifcause == 'Y':
                    cause_content = get_clean_data_en(itemc['cause_content'])
                    start, end = get_en_target(content_text, cause_content)
                    span_index.append([tagg + start, tagg + end])
                    taget_l[start] = 'B'
                    if start < end - 1:
                        for i in range(start+1, end):
                            taget_l[i] = 'I'
                target_data.extend(taget_l)#添加句子的原因标签
                tagg += len(content_l)

            example_dic['content_data'] = content_data
            example_dic['target_data'] = target_data
            example_dic['clause_len'] = clause_len
            example_dic['content_len'] = len(content_data)
            example_dic['emotion_word'] = emotion_content
            example_dic['emotion_len'] = len(emotion_content)
            example_dic['span_index'] = span_index
            return_data.append(example_dic)

        # for i in range(0, 3):
        #     dd = return_data[i]
        #     print(dd['content_data'])
        #     print(dd['target_data'])
        #     print('核对数据')
        return return_data


    def _read_ch_pkl(self, data_path, save_csv_path = None):
        #获取每个数据的属性
        """
        要获取文本数据， 情感数据， 原因的位置，以及要进行核对原因位置是否正确
        获取example的list列表
        """
        #[{'docID': 0}, 
        # {'name': 'happiness', 'value': '3'}, 
        # [{'key-words-begin': '0', 'keywords-length': '2', 'keyword': '激动', 'clauseID': 3, 'keyloc': 2}], 
        # [{'id': '1', 'type': 'v', 'begin': '43', 'length': '11', 'index': 1, 'cause_content': '接受并采纳过的我的建议', 'clauseID': 5}], 
        
        # [{'id': '1', 'cause': 'N', 'keywords': 'N', 'clauseID': 1, 'content': '河北省邢台钢铁有限公司的普通工人白金跃，', 'cause_content': '', 'dis': -2}, 
        # {'id': '2', 'cause': 'N', 'keywords': 'N', 'clauseID': 2, 'content': '拿着历年来国家各部委反馈给他的感谢信，', 'cause_content': '', 'dis': -1}, 
        # {'id': '3', 'cause': 'N', 'keywords': 'Y', 'clauseID': 3, 'content': '激动地对中新网记者说。', 'cause_content': '', 'dis': 0}, 
        # {'id': '4', 'cause': 'N', 'keywords': 'N', 'clauseID': 4, 'content': '“27年来，', 'cause_content': '', 'dis': 1}, 
        # {'id': '5', 'cause': 'Y', 'keywords': 'N', 'clauseID': 5, 'content': '国家公安部、国家工商总局、国家科学技术委员会科技部、卫生部、国家发展改革委员会等部委均接受并采纳过的我的建议', 'cause_content': '接受并采纳过的我的建议', 'dis': 2}]]
        
        return_data = []
        data = loadList(data_path)

        for index, item in enumerate(data):
            example_dic =dict()
            docID = item[0]['docID']
            emotion_loc = int(item[2][-1]['keyloc'])
            emotion_word = get_clean_data_ch(item[2][-1]['keyword']) #获取情去燥的情绪词语

            clause_info = item[4] #clause 信息

            emo_clause = clause_info[emotion_loc]['content']
            emotion_content = list(get_clean_data_ch(emo_clause))

            example_dic['docID'] = docID
            example_dic['emo_data'] = emotion_content
            
            content_data = []
            target_data = []
            clause_len = []
            span_index = []

            tagg = 0
            for indexc, itemc in enumerate(clause_info):
                content_text =get_clean_data_ch(itemc['content'])
                content_l = list(content_text)
                # content_l.append('[SEP]')#添加【SEP】字符
                content_data.extend(content_l) #添加子句的word
                clause_len.append(len(content_l))#获取子句的长度

                #获取类别标签
                taget_l = ['O'] * len(content_l)
                ifcause = itemc['cause']
                if ifcause == 'Y':
                    cause_content = get_clean_data_ch(itemc['cause_content'])
                    start, end = get_ch_target(content_text, cause_content)
                    span_index.append([tagg + start, tagg + end])
                    taget_l[start] = 'B'
                    if start < end - 1:
                        for i in range(start+1, end):
                            taget_l[i] = 'I'
                target_data.extend(taget_l)#添加句子的原因标签
                tagg += len(content_l)
                
            example_dic['content_data'] = content_data
            example_dic['emotion_word'] = list(emotion_word)
            example_dic['emotion_len'] = len(list(emotion_word))
            example_dic['target_data'] = target_data
            example_dic['clause_len'] = clause_len
            example_dic['content_len'] = len(content_data)
            example_dic['span_index'] = span_index
           
            return_data.append(example_dic)
        
        # for i in range(0, 3):
        #     dd = return_data[i]
        #     print(dd['content_data'])
        #     print(dd['target_data'])
        #     print('核对数据')

        return return_data


    def _read_sti_pkl(self, data_path, save_csv_path = None):
        #获取每个数据的属性
        """
        要获取文本数据， 情感数据， 原因的位置，以及要进行核对原因位置是否正确
        获取example的list列表
        """
        """
        s = "string. With. Punctuation?"
        table = str.maketrans({key: ' ' + key + ' ' for key in string.punctuation})
        print('table = ', table)
        new_s = s.translate(table)
        [
        data_dict['id'] =  str(len(all_data))
        data_dict['emotion'] = strr
        data_dict['text'] = text
        data_dict['cause'] = cause
       """
        return_data = []
        data = loadList(data_path)

        for index, item in enumerate(data):
            example_dic =dict()
            docID = item['id']
            
            # emo_clause = clause_info[emotion_loc]['content']
            emotion_content = item['emotion']

            example_dic['docID'] = docID
            emotion_word = get_clean_data_en(emotion_content).strip().split()

            content_text =get_clean_data_en(item['text'])
            content_l = content_text.split()
            taget_l = ['O'] * len(content_l)

            cause_content = get_clean_data_en(item['cause'])
            start, end = get_en_target(content_text, cause_content)
            span_index = [[start, end]]
            taget_l[start] = 'B'
            if start < end - 1:
                for i in range(start+1, end):
                    taget_l[i] = 'I'
            

            clause_len = []
            clause_len.append(len(content_l))

            example_dic['content_data'] = content_l
            example_dic['target_data'] = taget_l
            example_dic['clause_len'] = clause_len
            example_dic['content_len'] = len(content_l)
            example_dic['emotion_word'] = emotion_word
            example_dic['emotion_len'] = len(emotion_word)
            example_dic['span_index'] = span_index
            return_data.append(example_dic)

        for i in range(0, 3):
            dd = return_data[i]
            print(dd['content_data'])
            print(dd['target_data'])
            print('核对数据')
        return return_data


'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    content = pickle.load(pkl_file)
    pkl_file.close()
    return content

def get_en_target(para_text,cause):
    """
    获取原因内容
    和原因内容
    """
    text_token=para_text.split()
    cause_token=cause.split()

    start = -1 
    end = -1
    for i in range(0, len(text_token)):
        if text_token[i:i+len(cause_token)] == cause_token:
            start = i
            end = i + len(cause_token)
            return start, end
            
    if start == -1 or end == -1:
        print('text_token = ',text_token)
        print('cause_token = ',cause_token)
        raise ValueError("cause not in clause")
    return start, end


def get_ch_target(para_text,cause):
    """
    获取原因内容
    和原因内容
    """
    # print('para_text = ', para_text)
    # print('cause = ', cause)
    text_token = list(para_text)
    cause_token = list(cause)

    start = -1 
    end = -1
    for i in range(0, len(text_token)):
        if text_token[i:i+len(cause_token)] == cause_token:
            start = i
            end = i + len(cause_token)
            return start, end
            
    if start == -1 or end == -1:
        print('text_token = ',text_token)
        print('cause_token = ',cause_token)
        raise ValueError("cause not in clause")
    return start, end






    