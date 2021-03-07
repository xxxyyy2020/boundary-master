import torch
from collections import Counter
# from processors.utils_ner import get_entities
import pickle
import json
import numpy as np
import random
import sys
import os

tags = ['O','B','I']
'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent

def get_span(label_list):
    ifstart = False
    # print('label_list = ', label_list)
    span_l = []
    for i in range(len(label_list)):
        if label_list[i] != 0 and not ifstart:
            ifstart = True
            start = i 
        if label_list[i] == 0 and ifstart:
            ifstart = False
            end = i 
            span_l.append((start, end))
        
        if i == len(label_list) - 1 and ifstart:
            end = i + 1
            span_l.append((start, end))
            
    return span_l

def get_clause_label(pre_l, tru_l, data_len, exam_label):
    """
    对每一个文本获取子句级别的标签
    """
    pre = [0] * len(data_len)
    tru = [0] * len(data_len)
    predic_clause_len = len(pre_l)
    tru_clause_len = len(exam_label)

    if tru_clause_len > predic_clause_len:
        pad_len = tru_clause_len - predic_clause_len
        for i in range(pad_len):
            pre_l.append(0)
        tru_l = exam_label

    tag = 0
    for i in range(len(data_len)):
        prec = pre_l[tag: tag + int(data_len[i])]
        truc = tru_l[tag: tag + int(data_len[i])]
        if sum(prec) > 0:
            pre[i] = 1
        if sum(truc) > 0:
            tru[i] = 1
        tag += int(data_len[i])
    
    return pre, tru
    
    
def get_prf(pre_label_l, tru_label_l, examples):
    """
    pre_label_l: [list]
    tru_label_l: [list]
    data_len_c: [list]
    """ 
  
    
    assert len(pre_label_l) == len(tru_label_l) == len(examples)
    data_len_c = []
    tru_example_label = []
    for index, item in enumerate(examples):
        data_len_c.append(item.data_len_c)
        aa = [tags.index(tt) for tt in item.labels]
        tru_example_label.append(aa)

    p, r, f = 0, 0 ,0
    pre_span, tru_span, correct_span = 0, 0, 0
    pres, trus = [],[]

    for pre_label, tru_label, data_len, exam_label in zip(pre_label_l, tru_label_l, data_len_c, tru_example_label):

        assert len(pre_label) == len(tru_label)
        pre, tru = get_clause_label(pre_label, tru_label, data_len,exam_label)
        pres.extend(pre)
        trus.extend(tru)
        
        span_l_t = get_span(tru_label)
        span_l_p = get_span(pre_label)
        
        for indexp, itemp in enumerate(span_l_p):

            stp = itemp[0]
            enp = itemp[1]
            pre_span += 1
          
        for index, item_t in enumerate(span_l_t):
            stt = item_t[0]
            ent = item_t[1]
            tru_span += 1
           
            for indexp, itemp in enumerate(span_l_p):
                stp = itemp[0]
                enp = itemp[1]
                if stt == stp and ent==enp:
                    correct_span += 1
               
       
    p_p = p/len(pre_label_l)
    r_p = r/len(pre_label_l)
    f = 2*p_p * r_p /(p_p + r_p) if (p_p + r_p) > 0 else 0

    PS = correct_span / pre_span if pre_span > 0 else 0
    RS = correct_span / tru_span if tru_span > 0 else 0
    FS = 2* PS * RS / (PS+RS) if (PS+RS) > 0 else 0

    pre = np.sum(np.array(pres))
    tru = np.sum(np.array(trus))
    correct = np.sum(np.multiply(np.array(pres), np.array(trus)))
    
    p_c = 1.0 * correct / pre if pre > 0 else 0
    r_c = 1.0 * correct / tru if tru > 0 else 0
    f_c = 2 * p_c * r_c/ (p_c + r_c) if (p_c + r_c) >0 else 0

    result = {'span_p': np.around(PS, decimals=4),'span_r': np.around(RS, decimals=4), 'span_f': np.around(FS, decimals=4), 'p_c': np.around(p_c, decimals=4), 'p_r': np.around(r_c, decimals=4), 'f_c': np.around(f_c, decimals=4)}
    result_span = "span: {} {} {}".format(np.around(PS, decimals=4), np.around(RS, decimals=4), np.around(FS, decimals=4))
    result_c = "clause: {} {} {}".format(np.around(p_c, decimals=4),  np.around(r_c, decimals=4), np.around(f_c, decimals=4))
    
    # print('\n')
    # print(result_span)
    # print(result_c)

    return result