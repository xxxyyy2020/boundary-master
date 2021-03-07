## Load the pre-trained file

load the pre-trained bert base models bert_based_ch and bert_base_en from  https://pan.baidu.com/s/1MvrsuuvPfKDcri9vUKwWOg and the extraction code is: mepv. 
and then put them in current file.

## train the models
python train.py \
--model_encdec bert2crf  --max_seq_length 500  --answer_seq_len 3 --save_name  bert2crf_CH --data_type ch


python train.py \
--model_encdec bert2crf  --max_seq_length 500  --answer_seq_len 3 --save_name  bert2crf  --data_type ch









