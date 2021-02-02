import argparse
import os
path = os.getcwd()

def get_argparse():

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--root_path", default= path, type=str, 
                        help="The root path: ")
    parser.add_argument("--task_name", default='eca', type=str, 
                        help="The name of the task to train selected in the list: ")
    parser.add_argument("--data_dir", default=os.path.join(path,'dataset_eca'), type=str,
                        help="The generated dataset in each split.", )
    parser.add_argument("--results_dir", default=os.path.join(path,'results_file'), type=str,
                        help="The result file.", )
    parser.add_argument("--decoder_num_layers", default=2, type=int,  help="The number of the decoder layers.")
    
    parser.add_argument("--train_batch_size", default=5, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=5, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--split_times", default=3, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--Gpu_num", default=0, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--data_type", default='sti', type=str,
                        help="sti")
    parser.add_argument("--model_encdec", default='bert2soft', type=str, 
                        help="Model type selected in the list: 'bert2crf', 'bert2gru' , 'bert2soft'")
#     parser.add_argument("--att_type", default='Att', type=str, 
#                         help="Model type selected in the list: CoAtt, Att ")
    parser.add_argument("--label_embedding_size", default=50, type=int, 
                        help="")
    parser.add_argument("--dropout", default=0.5, type=float, help=".")
    parser.add_argument("--encoder_hidden_size", default=384, type=float, help=".")
    parser.add_argument("--decoder_hidden_size", default=300, type=float, help=".")
    parser.add_argument("--attention_hidden_size", default=300, type=float, help=".")
    parser.add_argument("--label_size", default=3, type=int, 
                        help="the number pf the numbers of label ['O', 'B', 'I'] ")

    parser.add_argument("--model_type", default='bert', type=str, 
                        help="Model type selected in the list: ")
#     parser.add_argument("--model_name_or_path", default=os.path.join(path,'bert_base_en'), type=str,
#                         help="Path to pre-trained model or shortcut name selected in the list: bert_base_chinese" )
    parser.add_argument("--output_dir", default=os.path.join(path,'output_'), type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--dataset_eca_raw", default=os.path.join(path,'dataset_eca_raw'), type=str, 
                        help="the raw data set file.", )
    # Other parameters
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded 150.", )
    parser.add_argument("--eval_max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=True,
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", default=True,
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    # adversarial training
    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=1e-2, type=float,
                        help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=30,
                        help="Log every X updates steps. 50")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", default=False,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
                             
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--answer_seq_len", type=int, default= 5, help="For distributed training: local_rank")

    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--save_name", type=str, default="sti", help="For distant debugging.")

    parser.add_argument("--soft_label", type=bool, default=True, help=".")

    parser.add_argument("--emotion", action="store_true",
                        help="Overwrite the cached training and evaluation sets")

    return parser
