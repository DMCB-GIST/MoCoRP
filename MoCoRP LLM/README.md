# Model training and inference
## Requirements
- python 3.10.13
- cuda 11.8
- torch 2.1.0
- transformers 4.45.2
- ubuntu 22.04.6

## Environment setup
```bash
conda create -n MoCoRP LLM python=3.10 -y
conda activate MoCoRP LLM
pip install -r requirements.txt
```

## MoCoRP LLM
### Prerequisite
Train NLI expert using dialogue NLI dataset
```bash
CUDA_VISIBLE_DEVICES=0 python main_nli_expert.py \
    --model_checkpoint FacebookAI/roberta-large \
    --save_dir models/dnli/nli_expert \
    --do_train
```

### Train prior LLM
Train prior LLM through SFT and DPO
```bash
# SFT
export num_train_epochs=5
export batch_size=16
export lr=5e-5
export lora_r=8
export lora_alpha=16
torchrun --nproc_per_node=8 train_sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --output_dir outputs/sft_qwen2-0.5b_ep${num_train_epochs}_bs${batch_size}_lr${lr}_r${lora_r}_al${lora_alpha} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_checkpointing True \
    --optim paged_adamw_32bit \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.001 \
    --max_grad_norm 1.0 \
    --use_peft \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_target_modules all-linear \
    --torch_dtype bfloat16 \
    --bf16 True \
    --tf32 True \
    --eval_strategy epoch \
    --save_strategy epoch \
    --max_seq_length 4096 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False

# DPO
export model_name=
export checkpoint=
export num_train_epochs=1
export batch_size=8
export lr=5e-7
export lora_r=8
export lora_alpha=16
torchrun --nproc_per_node=8 train_dpo.py \
    --model_name_or_path ../sft/outputs/${model_name}/${checkpoint} \
    --output_dir outputs/dpo_qwen2-0.5b_ep${num_train_epochs}_bs${batch_size}_lr${lr}_r${lora_r}_al${lora_alpha}-${model_name} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --precompute_ref_log_probs True \
    --gradient_checkpointing True \
    --optim paged_adamw_32bit \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.001 \
    --max_grad_norm 1.0 \
    --use_peft \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_target_modules all-linear \
    --torch_dtype bfloat16 \
    --bf16 True \
    --tf32 True \
    --eval_strategy epoch \
    --save_strategy epoch \
    --max_length 4096 \
    --max_prompt_length 1024 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False
```


### Generate prior NLI relations
```bash
# generate valid dataset response
export sft_model_name=
export sft_checkpoint=
export model_name=
export checkpoint=
torchrun --nproc_per_node=8 test_dpo.py \
    --model_name_or_path outputs/${model_name}/${checkpoint} \
    --sft_model_name_or_path ../sft/outputs/${sft_model_name}/${sft_checkpoint} \
    --output_dir outputs/${model_name} \
    --per_device_eval_batch_size 1 \
    --torch_dtype float16 \
    --fp16 True \
    --tf32 True \
    --max_length 4096 \
    --max_prompt_length 1024 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True

# generate train dataset response
export sft_model_name=
export sft_checkpoint=
export model_name=
export checkpoint=
torchrun --nproc_per_node=8 test_dpo.py \
    --model_name_or_path outputs/${model_name}/${checkpoint} \
    --sft_model_name_or_path ../sft/outputs/${sft_model_name}/${sft_checkpoint} \
    --output_dir outputs/${model_name} \
    --per_device_eval_batch_size 1 \
    --torch_dtype float16 \
    --fp16 True \
    --tf32 True \
    --max_length 4096 \
    --max_prompt_length 1024 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --generate_train True

# Compute prior NLI relations
# compute valid dataset relations
CUDA_VISIBLE_DEVICES=0 python compute_relations.py \
    --model_checkpoint outputs/dnli/nli_expert/checkpoint-1-4846 \
    --task convai2 \
    --data_path data/convai2/valid_self_original.txt \
    --nli_data_path $nli_data_path \
    --prediction_input_path $prediction_input_path

# compute train dataset relations
CUDA_VISIBLE_DEVICES=0 python compute_relations.py \
    --model_checkpoint outputs/dnli/nli_expert/checkpoint-1-4846 \
    --task convai2 \
    --data_path data/convai2/train_self_original.txt \
    --nli_data_path $nli_data_path \
    --prediction_input_path $prediction_input_path
```



### Train posterior LLM
Train posterior LLM through SFT and DPO
```bash
# SFT
export num_train_epochs=5
export batch_size=16
export lr=5e-5
export lora_r=8
export lora_alpha=16
torchrun --nproc_per_node=8 train_sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --output_dir outputs/sft_post_qwen2-0.5b_ep${num_train_epochs}_bs${batch_size}_lr${lr}_r${lora_r}_al${lora_alpha} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_checkpointing True \
    --optim paged_adamw_32bit \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.001 \
    --max_grad_norm 1.0 \
    --use_peft \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_target_modules all-linear \
    --torch_dtype bfloat16 \
    --bf16 True \
    --tf32 True \
    --eval_strategy epoch \
    --save_strategy epoch \
    --max_seq_length 4096 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --nli_data_dir $nli_data_dir

# DPO
export model_name=
export checkpoint=
export num_train_epochs=1
export batch_size=8
export lr=5e-7
export lora_r=8
export lora_alpha=16
torchrun --nproc_per_node=8 train_dpo.py \
    --model_name_or_path ../sft/outputs/${model_name}/${checkpoint} \
    --output_dir outputs/dpo_post_qwen2-0.5b_ep${num_train_epochs}_bs${batch_size}_lr${lr}_r${lora_r}_al${lora_alpha}-${model_name} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --precompute_ref_log_probs True \
    --gradient_checkpointing True \
    --optim paged_adamw_32bit \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.001 \
    --max_grad_norm 1.0 \
    --use_peft \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_target_modules all-linear \
    --torch_dtype bfloat16 \
    --bf16 True \
    --tf32 True \
    --eval_strategy epoch \
    --save_strategy epoch \
    --max_length 4096 \
    --max_prompt_length 1024 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --nli_data_dir $nli_data_dir
```
