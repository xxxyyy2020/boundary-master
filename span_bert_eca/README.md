
## The pre-trained file 
Downloads the pre-trained file and then put it to the file: bert_base_en 

SpanBERT (base & cased): 12-layer, 768-hidden, 12-heads , 110M parameters 
(https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz)

## Train the models:

python train.py \
  --model spanbert-base-cased \
  --train_file train-v1.1.json \
  --dev_file dev-v1.1.json \
  --train_batch_size 32 \
  --eval_batch_size 32  \
  --learning_rate 2e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_metric f1 \
  --output_dir squad_output \
  --fp16
