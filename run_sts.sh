#!/bin/bash
# model=${MODEL:-/root/.cache/huggingface/hub/models--princeton-nlp--sup-simcse-roberta-base/snapshots/4bf73c6b5df517f74188c5e9ec159b2208c89c08}  # pre-trained model
# model=${MODEL:-/root/.cache/huggingface/hub/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68}  # pre-trained model
# model=${MODEL:-/root/.cache/huggingface/hub/models--roberta-large/snapshots/716877d372b884cad6d419d828bac6c85b3b18d9}
# model=${MODEL:-/root/.cache/huggingface/hub/models--princeton-nlp--sup-simcse-roberta-large/snapshots/96d164d9950b72f4ce179cb1eb3414de0910953f}
model=${MODEL=princeton-nlp/sup-simcse-roberta-base}
encoding=${ENCODER_TYPE:-cross_encoder}  # cross_encoder
lr=${LR:-3e-5}  # learning rate
wd=${WD:-0.1}  # weight decay
transform=${TRANSFORM:-True}  # whether to use an additional linear layer after the encoder, True for optimal performance
objective=${OBJECTIVE:-mse}  # mse
triencoder_head=${TRIENCODER_HEAD:-None}  # hadamard, concat (set for tri_encoder)
seed=${SEED:-42}
output_dir=${OUTPUT_DIR:-output}
config=enc_${encoding}__lr_${lr}__wd_${wd}__trans_${transform}__obj_${objective}__tri_${triencoder_head}__s_${seed}
train_file=${TRAIN_FILE:-data/csts_train.csv}
eval_file=${EVAL_FILE:-data/csts_validation.csv}
test_file=${TEST_FILE:-data/csts_test.csv}

python run_sts.py \
  --output_dir "${output_dir}/${model//\//__}/${config}" \
  --model_name_or_path ${model} \
  --objective ${objective} \
  --encoding_type ${encoding} \
  --pooler_type cls \
  --freeze_encoder False \
  --transform ${transform} \
  --triencoder_head ${triencoder_head} \
  --max_seq_length 128 \
  --train_file ${train_file} \
  --validation_file ${eval_file} \
  --test_file ${test_file} \
  --condition_only False \
  --sentences_only False \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate ${lr} \
  --weight_decay ${wd} \
  --max_grad_norm 0.0 \
  --num_train_epochs 6 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.1 \
  --log_level info \
  --disable_tqdm True \
  --save_strategy epoch \
  --save_total_limit 1 \
  --seed ${seed} \
  --data_seed ${seed} \
  --fp16 True \
  --log_time_interval 15