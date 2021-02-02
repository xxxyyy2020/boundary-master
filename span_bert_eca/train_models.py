# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import time
import shutil
import re
import string
import sys
from io import open
import numpy as np
import torch
import torch.nn as nn
import copy
# from torch.utils.data import DataLoader, TensorDataset
from tools.common import seed_everything
from tools.progressbar import ProgressBar
from process_data.func import loadList, saveList

# from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.file_utils import  WEIGHTS_NAME, CONFIG_NAME
# PYTORCH_PRETRAINED_BERT_CACHE: /home/MHISS/liqiang/.pytorch_pretrained_bert
# WEIGHTS_NAME:pytorch_model.bin
# CONFIG_NAME：config.json
from process_data.eca_seq import convert_examples_to_features 
# from process_data.eca_seq import eca_processors as processors, batch_generator # ner_processors = {"cner": CnerProcessor,'cluener':CluenerProcessor, 'eca':ECAProcessor}
from process_data.eca_seq import eca_processors as processors, batch_generator, bert_extract_item, extract_multi_item # ner_processors = {"cner": CnerProcessor,'cluener':CluenerProcessor, 'eca':ECAProcessor}

# from processors_eca.eca_seq import collate_fn
# from pytorch_pretrained_bert.modeling import
from models.SpanbertForEca_new import  Bert2Crf, Bert2Gru, Bert_softmax, Bert_Multi_Point
# Bert2Linear_span_M
from pytorch_pretrained_bert.optimizer.BertAdam import BertAdam, warmup_linear
from pytorch_pretrained_bert.optimizer.AdamW import AdamW
from pytorch_pretrained_bert.lr_scheduler import get_linear_schedule_with_warmup

from pytorch_pretrained_bert.tokenization import (BasicTokenizer, whitespace_tokenize)
from process_data.utils_eca import EcaTokenizer
from funting_args import get_argparse
from metrics.eca_metrics import get_prf #

args = get_argparse().parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Gpu_num)

args.model_name_or_path = os.path.join(os.getcwd(),'span_bert_cause')

if args.model_encdec == 'bert2crf':
    MODEL_CLASSES = { 'bert': Bert2Crf}
if args.model_encdec == 'multi2point':
    MODEL_CLASSES = { 'bert': Bert_Multi_Point}
elif args.model_encdec == 'bert2soft':
    MODEL_CLASSES = { 'bert': Bert_softmax}
elif args.model_encdec == 'bert2gru':
    MODEL_CLASSES = { 'bert':  Bert2Gru }

#创建example
#创建feature
#创建[example]
#将example 转换为feature
logger = logging.getLogger(__name__)

# 加载数据，并转换为tensor类型，还进行tokenizer
def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    processor = processors[task]()
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels() #[B I O]
    if data_type == 'train':
        examples = processor.get_train_examples(args.data_dir) #获取数据，并且 增加了将数据存储为csv文件
    elif data_type == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.train_max_seq_length, #需要测试一下两部分数据的最大长度是什么
                                            # pad on the left for xlnet
                                            pad_token=0,
                                            pad_token_segment_id= 0,
                                            )
    return features


def train(args, train_features, model, tokenizer, use_crf):
    """ Train the model """
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())

    if args.model_encdec == 'bert2crf':
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate},

                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}
            ]

    elif args.model_encdec == 'bert2gru':
        gru_param_optimizer = list(model.decoder.named_parameters())
        linear_param_optimizer = list(model.clsdense.named_parameters())
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in gru_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in gru_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate},

                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}]
    
    elif args.model_encdec == 'bert2soft':
        # gru_param_optimizer = list(model.decoder.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}
            ]
    
    elif args.model_encdec == 'multi2point':
        # gru_param_optimizer = list(model.decoder.named_parameters())
        linear_param_optimizer = list(model.pointer.named_parameters())
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}
            ]
    
    elif args.model_encdec == 'multi2soft':
        # gru_param_optimizer = list(model.decoder.named_parameters())
        qa_outputs_param_optimizer = list(model.qa_outputs.named_parameters())
        num_outputs_param_optimizer = list(model.num_outputs.named_parameters())
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in qa_outputs_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in qa_outputs_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate},

                {'params': [p for n, p in num_outputs_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in num_outputs_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}
            ]
        
    t_total = len(train_features)//args.train_batch_size * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss  = 0.0, 0.0
    pre_result = {}
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    total_step = 0
    best_spanf = -1

    test_results = {}
    for ep in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_features)//args.train_batch_size, desc='Training')
        step= 0
        for batch in batch_generator(features = train_features, batch_size=args.train_batch_size,  use_crf = use_crf, answer_seq_len = args.answer_seq_len):
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids, batch_multi_span_label, batch_context_mask, batch_start_position, batch_end_position, batch_raw_labels, _, batch_example = batch
           
            model.train()
            if args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2gru' or args.model_encdec == 'bert2soft':
                batch_inputs = tuple(t.to(args.device) for t in batch[0:6])
                inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], "context_mask": batch_inputs[5], "labels": batch_inputs[3], "testing":False}
               
            elif args.model_encdec == 'multi2point':
                batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
                inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], "span_label":batch_inputs[4], "testing":False}
            
            elif args.model_encdec == 'multi2soft':
                batch_inputs = tuple(t.to(args.device) for t in batch[0:8])
                inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2],  "start_positions": batch_inputs[6], "end_positions": batch_inputs[7], "testing":False}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss.backward()
            pbar(step, {'loss': loss.item()})
            step += 1
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # optimizer.step()
                # optimizer.zero_grad()
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print("start evalue")
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args = args, model = model, tokenizer = tokenizer,  prefix="dev", use_crf = use_crf)
                    span_f = results['span_f'] #在span级别f的值

                    if span_f > best_spanf:
                        output_dir = os.path.join(args.output_dir, "checkpoint-bestf")
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                            print('删除文件夹：',args.output_dir)
                            print('eval_result:',results)
                            test_results = evaluate(args = args, model = model, tokenizer = tokenizer,  prefix="test", use_crf = use_crf)
                            print('test_result:', test_results)
                        best_spanf = span_f
                        os.makedirs(output_dir)
                        # print('dir = ', output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_dir, CONFIG_NAME)
                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_dir)
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

        # train_history.append(loss)
        np.random.seed()
        np.random.shuffle(train_features)
        logger.info("\n")
        # if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step, test_results



def evaluate(args, model, tokenizer, prefix="dev", use_crf = False):
    # metric = get_prf(args.id2label, markup=args.markup)

    eval_features = load_and_cache_examples(args, args.data_type, tokenizer, data_type=prefix)
    processor = processors[args.data_type]()
   
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    # nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_features), desc="Evaluating" + prefix)
    if isinstance(model, nn.DataParallel):
        model = model.module
    
    pre_labels, tru_labels, eval_examples  =[],  [], [] #统计整个eval数据的 标签
    # pbar = ProgressBar(n_total=len(eval_features)//args.train_batch_size, desc='Training')
    step = 0
    for batch in batch_generator(features = eval_features, batch_size=args.train_batch_size,  use_crf = use_crf, answer_seq_len = args.answer_seq_len):

        batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids, batch_multi_span_label, batch_context_mask, batch_start_position, batch_end_position, batch_raw_labels, _, batch_example = batch
        # batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids, batch_span_label, batch_context_mask,  batch_start_position, batch_end_position, batch_raw_labels, _, batch_example = batch
        # print('aa = ', batch_input_ids )
        model.eval()
        if args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2gru' or args.model_encdec == 'bert2soft':
                batch_inputs = tuple(t.to(args.device) for t in batch[0:6])
                inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], "context_mask": batch_inputs[5], "testing":True}
                # inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], "context_mask": batch_inputs[5], "labels": batch_inputs[3], "span_label":batch_inputs[4], "start_positions": batch_inputs[6], "end_positions": batch_inputs[7],"testing":False}
            
        elif args.model_encdec == 'multi2point':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2],  "testing":True}
        
        elif args.model_encdec == 'multi2soft':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:8])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2],  "testing":True}

        outputs = model(**inputs)
        eval_examples.extend(batch_example)
        out_label_ids = batch[8].tolist() #真实标签
        batch_lens = torch.sum(batch_context_mask, -1).cpu().numpy().tolist()

        if args.model_encdec == 'bert2crf':
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()

            for len_doc, cu_tags, cu_trus, exam in zip(batch_lens, tags, out_label_ids, batch_example):
                emotion_len = exam.emotion_len
                pre_labels.append(cu_tags[1: len_doc + 1])
                tru_labels.append(cu_trus[1: len_doc + 1])

        elif args.model_encdec == 'multi2point':

            start_label, end_label = outputs #[batch, ans_len]
            start_label, end_label = start_label.cpu().numpy().tolist(), end_label.cpu().numpy().tolist()

            pres_batch = []
            for s_num, e_num in zip(start_label, end_label):
                pre_tag = [0] * args.eval_max_seq_length
                for s, e in zip(s_num, e_num):
                    if s < e - 1:
                        pre_tag[s] = 1
                        pre_tag[s+1:e] = [2] * (e - s -1)
                    elif s == e -1:
                        pre_tag[s] = 1
                pres_batch.append(pre_tag)
            
            for len_doc, cu_tags, cu_trus, exam in zip(batch_lens, pres_batch, out_label_ids, batch_example):
                emotion_len = exam.emotion_len
                pre_labels.append(cu_tags[1: len_doc + 1])
                tru_labels.append(cu_trus[1: len_doc + 1])

        elif args.model_encdec == 'multi2soft':

            start_logits, end_logits, num_logits = outputs[:3]
            pres_batch = extract_multi_item(start_logits, end_logits, num_logits)
            for len_doc, cu_tags, cu_trus, exam in zip(batch_lens, pres_batch, out_label_ids, batch_example):
                emotion_len = exam.emotion_len
                pre_labels.append(cu_tags[1: len_doc + 1])
                tru_labels.append(cu_trus[1: len_doc + 1])
            
        elif args.model_encdec == 'bert2gru' or args.model_encdec == 'bert2soft':
            tags = outputs.cpu().numpy()
            tags = tags.tolist()
            # out_label_ids = batch[5].tolist() #真实标签
            # pre_labels =[list(p[i]) for i in range(p.shape[0])]
            for len_doc, cu_tags, cu_trus, exam in zip(batch_lens, tags, out_label_ids, batch_example):
                pre_labels.append(cu_tags[1: len_doc + 1])
                tru_labels.append(cu_trus[1: len_doc + 1])

        step += 1
        pbar(step)
    logger.info("\n")
    results = get_prf(pre_labels, tru_labels, eval_examples)
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    return results


def train_model():
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1

    if args.model_encdec == 'bert2gru' or args.model_encdec == 'bert2soft' or args.model_encdec == 'multi2point' or args.model_encdec == 'multi2soft':
        use_crf = False
    elif args.model_encdec == 'bert2crf':
        use_crf = True

    args.device = device
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.data_type = args.data_type.lower() 
    if args.data_type not in processors: 
        raise ValueError("Task not found: %s" % (args.data_type))
    processor = processors[args.data_type]()
    label_list = processor.get_labels()
    
    args.id2label = {i: label for i, label in enumerate(label_list)} #获取的是字典{0: O, 1: B, 2: I}
    args.label2id = {label: i for i, label in enumerate(label_list)} #获取的是字典{O:0, B:1, I: 2}
    num_labels = len(label_list) # 标签的个数

    args.model_type = args.model_type.lower()
    
    tokenizer = EcaTokenizer.from_pretrained( args.model_name_or_path, do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,
                                               )
    model_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_name_or_path, cache_dir=None)
    model.to(device)
    
    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_features = load_and_cache_examples(args, args.data_type, tokenizer, data_type='train')
    global_step, tr_loss, metrics = train(args, train_features, model, tokenizer, use_crf = use_crf)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)  

    return metrics


def main():
    """
    args:
    """
    if not os.path.exists(args.results_dir): #如果不存在进行创建
        os.mkdir(args.results_dir)

    list_result = []
    # print(args.split_times)
    out_current_path = copy.deepcopy(args.output_dir) 

    if args.data_type == 'ch':
        test_data = loadList(os.path.join(os.getcwd(), 'eca_data/eca_ch/split_data_fold/eca_ch_test_example.pkl'))
    elif args.data_type == 'en':
        test_data = loadList(os.path.join(os.getcwd(), 'eca_data/eca_en/split_data_fold/eca_en_test_example.pkl'))
    elif args.data_type == 'sti':
        test_data = loadList(os.path.join(os.getcwd(), 'eca_data/eca_sti/split_data_fold/eca_sti_test_example.pkl'))

    for i in range(args.split_times):
        print("*****************************split_times:{}*******************".format(i))
        args.output_dir = args.output_dir + '{}_{}_{}_{}_{}'.format(args.data_type, args.model_encdec, args.Gpu_num, i, args.save_name) #输出模型文件的位置
        if os.path.exists(args.output_dir): #是否存在输出文件，如果不存在进行创建
            shutil.rmtree(args.output_dir)
        os.mkdir(args.output_dir)
      
      
        data_current_path = copy.deepcopy(args.data_dir) 
        args.data_dir = args.data_dir  + '{}_{}_{}_{}_{}'.format(args.data_type, args.model_encdec, args.Gpu_num, i, args.save_name)  
        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
        os.mkdir(args.data_dir)
        
        
        if args.data_type == 'sti':
            train_data = []
            for eee in range(10):
                dev_each = loadList(os.path.join(os.getcwd(), 'eca_data/eca_sti/split_data_fold/eca_sti_dev{}_example.pkl'.format(eee)))
                if eee != i:
                    train_data.extend(dev_each)
                else:
                    dev_data = dev_each


        saveList(train_data, os.path.join(args.data_dir, 'eca_train.pkl'))
        saveList(test_data, os.path.join(args.data_dir, 'eca_test.pkl'))
        saveList(dev_data, os.path.join(args.data_dir, 'eca_dev.pkl'))

   
        metrics = train_model()
        list_result.append(metrics)
        print('bert_results_{} = {}\n'.format(args.data_type, metrics))

        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)

        
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
            print('remove files：', args.output_dir)
            
    
        args.output_dir = out_current_path
        args.data_dir = data_current_path

if __name__ == "__main__":
    main()
