## Load the pre-trained file

load the pre-trained bert base models bert_based_ch and bert_base_en from  https://pan.baidu.com/s/1MvrsuuvPfKDcri9vUKwWOg and the extraction code is: mepv. 
and then put them in current file.

##Train the models

## Split the data randomly

python  dataset/eca_ch/split_data_fold/split_data.py \
python  dataset/eca_en/split_data_fold/split_data.py 

## On Chinese dataset
python train.py \
--model_encdec bert2crf  --max_seq_length 500  --answer_seq_len 3 --save_name  bert2crf_CH --data_type ch


python train.py \
--model_encdec multi2point  --max_seq_length 500  --answer_seq_len 3 --save_name  multi2point_CH  --data_type ch


python train.py \
--model_encdec bert2soft  --max_seq_length 500  --answer_seq_len 3 --save_name  bert2soft_CH  --data_type ch


python train.py \
--model_encdec bert2gru  --max_seq_length 500  --answer_seq_len 3 --save_name  bert2gru_CH  --data_type ch




## On English dataset

python train.py \
--model_encdec bert2crf  --max_seq_length 200  --answer_seq_len 3 --save_name  bert2crf_EN --data_type sti


python train.py \
--model_encdec multi2point   --max_seq_length 200   --answer_seq_len 3 --save_name  bert2crf_EN  --data_type sti


python train.py \
--model_encdec bert2soft   --max_seq_length 200   --answer_seq_len 3 --save_name  bert2crf_EN  --data_type sti


python train.py \
--model_encdec bert2gru   --max_seq_length 200   --answer_seq_len 3 --save_name  bert2crf_EN  --data_type sti





