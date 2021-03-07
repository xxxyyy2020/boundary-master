
## The pre-trained file 
Downloads the pre-trained file and then put it to the file: bert_base_en 

SpanBERT (base & cased): 12-layer, 768-hidden, 12-heads , 110M parameters 
(https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz)

## Train the models:

python train_models.py \
  --model_encdec bert2crf 
  --save_name bert2crf
  

python train_models.py \
  --model_encdec multi2point 
  --save_name multi2point
  

python bert2soft.py \
  --model_encdec bert2soft 
  --save_name bert2soft
  


python bert2gru.py \
  --model_encdec bert2gru 
  --save_name bert2gru
  


