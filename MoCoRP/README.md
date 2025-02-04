# Model training and inference
## Requirements
- python 3.8.19
- cuda 11.3
- torch 1.12.0
- transformers 4.36.1
- ubuntu 18.04.6

## Environment setup
```bash
conda create -n MoCoRP python=3.8 -y
conda activate MoCoRP
pip install -r requirements.txt
```

## MoCoRP
### Prerequisite
Train NLI expert using dialogue NLI dataset
```bash
CUDA_VISIBLE_DEVICES=0 python main_nli_expert.py \
    --model_checkpoint FacebookAI/roberta-large \
    --save_dir models/dnli/nli_expert \
    --do_train
```

### Compute NLI relations (ConvAI2)
```bash
CUDA_VISIBLE_DEVICES=0 python compute_relations.py \
    --model_checkpoint outputs/dnli/nli_expert/checkpoint-1-4846 \
    --task convai2 \
    --data_path data/convai2/train_self_original.txt \
    --nli_data_path $nli_data_path
```


### Train & evaluate MoCoRP (ConvAI2)
```bash
# Train
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task convai2 \
    --original True \
    --dialog_data_dir data/convai2 \
    --model_name_or_path facebook/bart-large \
    --output_dir outputs/convai2/mocorp \
    --nli_data_dir data/convai2/ \
    --dnli_data_path data/dialogue_nli\
    --max_history 7 \
    --num_candidates 4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 8e-6 \
    --rp_coef 0.1 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --predict_with_generate False

# Eval
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task convai2 \
    --original True \
    --dialog_data_dir data/convai2 \
    --model_name_or_path outputs/convai2/mocorp/checkpoint-101019 \
    --output_dir outputs/convai2/mocorp/ \
    --nli_data_dir data/convai2/ \
    --nli_model_name_or_path outputs/dnli/nli_expert/checkpoint-1-4846 \
    --max_history 7 \
    --num_candidates 4 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate True
```





### Compute NLI relations (MPChat)
```bash
CUDA_VISIBLE_DEVICES=0 python compute_relations.py \
    --model_checkpoint outputs/dnli/nli_expert/checkpoint-1-4846 \
    --task mpchat \
    --data_path data/mpchat/train_mpchat_nrp.json \
    --nli_data_path $nli_data_path

CUDA_VISIBLE_DEVICES=0 python compute_relations.py \
    --model_checkpoint outputs/dnli/nli_expert/checkpoint-1-4846 \
    --task mpchat \
    --data_path data/mpchat/valid_mpchat_nrp.json \
    --nli_data_path $nli_data_path

```

### Train & evaluate MoCoRP (MPChat)
```bash
# Train
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task mpchat \
    --dialog_data_dir data/mpchat \
    --model_name_or_path facebook/bart-large \
    --output_dir outputs/mpchat/mocorp \
    --nli_data_dir data/mpchat/ \
    --max_history 7 \
    --num_candidates 4 \
    --num_train_epochs 5 \
    --early_stop_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --rp_coef 0.1 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --predict_with_generate False

# Eval
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task mpchat \
    --dialog_data_dir data/mpchat \
    --model_name_or_path outputs/mpchat/mocorp/checkpoint-28770 \
    --output_dir outputs/mpchat/mocorp/ \
    --nli_data_dir data/mpchat/ \
    --nli_model_name_or_path outputs/dnli/nli_expert/checkpoint-1-4846 \
    --max_history 7 \
    --num_candidates 4 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate True
```
