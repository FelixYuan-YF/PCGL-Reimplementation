type=homographic
# type=homophonic
# model=Qwen2.5-3B-Instruct
model=Qwen2.5-3B
template=qwen

python process_dataset.py --output_dir ./data --repeat_times 10

CUDA_VISIBLE_DEVICES=3 python vllm_infer.py \
  --model_name_or_path ./LLM/${model} \
  --template ${template} \
  --dataset ${type}_cn_generate \
  --save_name ./${model}-${type}-cn.json \
  --top_p 0.95 \
  --top_k 5 \
  --temperature 0.95 \

python reconstruct.py \
  --raw_file ./data/${type}_cn.json \
  --generate_file ./${model}-${type}-cn.json \
  --output_file ./data/dpo_${type}_cn_1.json

CUDA_VISIBLE_DEVICES=3 llamafactory-cli train \
    --stage dpo \
    --do_train True \
    --model_name_or_path ./LLM/${model} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset dpo_${type}_cn_1 \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir ./saves/${model}/${type}/stage1 \
    --bf16 True \
    --plot_loss True \
    --overwrite_output_dir True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --pref_beta 0.5 \
    --pref_ftx 0 \
    --pref_loss dpop \
    --dpop_lambda 0.5 \
    --top_p 0.95 \
    --top_k 5 \
    --temperature 0.95

CUDA_VISIBLE_DEVICES=3 python vllm_infer.py \
  --model_name_or_path ./LLM/${model} \
  --adapter_name_or_path ./saves/${model}/${type}/stage1 \
  --template ${template} \
  --dataset ${type}_cn_generate \
  --max_new_tokens 300 \
  --save_name ./${model}-${type}-cn.json \
  --top_p 0.95 \
  --top_k 5 \
  --temperature 0.95

python reconstruct.py \
  --raw_file ./data/${type}_cn.json \
  --generate_file ./${model}-${type}-cn.json \
  --output_file ./dpo_${type}_cn_2.json

CUDA_VISIBLE_DEVICES=3 llamafactory-cli train \
    --stage dpo \
    --do_train True \
    --model_name_or_path ./LLM/${model} \
    --adapter_name_or_path ./saves/${model}/${type}/stage1 \
    --model_name_or_path ./LLM/${model} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset dpo_${type}_cn_2 \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir ./saves/${model}/${type}/stage2 \
    --bf16 True \
    --plot_loss True \
    --overwrite_output_dir True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --pref_beta 0.5 \
    --pref_ftx 0 \
    --pref_loss humor \
    --humor_gamma 0.5 \
    --top_p 0.95 \
    --top_k 5 \
    --temperature 0.95

CUDA_VISIBLE_DEVICES=3 llamafactory-cli chat \
    --model_name_or_path ./LLM/${model} \
    --adapter_name_or_path ./saves/${model}/${type}/stage2 \
    --template qwen \
    --infer_backend huggingface \
    --trust_remote_code True
