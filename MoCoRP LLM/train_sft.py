import os
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import (
    SFTConfig,
    SFTTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM,
)
from peft import PeftConfig, PeftModel

from utils.utils_sft import build_sft_dataset


@dataclass
class DataArguments:
    dialog_data_dir: Optional[str] = field(default="../data", metadata={"help": "Path to the training data."})
    nli_data_dir: Optional[str] = field(default=None, metadata={"help": "Path to the training data."})
    original: Optional[bool] = True

    max_history: Optional[int] = field(default=7, metadata={"help": ""})
    num_candidates: Optional[int] = field(default=4, metadata={"help": ""})
    max_candidates: Optional[int] = 20

    max_new_tokens: Optional[int] = field(default=100)

    def __post_init__(self):
        self.preprocess_fct = build_sft_dataset
        self.train_data_path = os.path.join(self.dialog_data_dir, f"train_self_{'original' if self.original else 'revised'}.txt")
        self.valid_data_path = os.path.join(self.dialog_data_dir, f"valid_self_{'original' if self.original else 'revised'}.txt")
        self.test_data_path = os.path.join(self.dialog_data_dir, f"valid_self_{'original' if self.original else 'revised'}.txt")
        self.train_nli_data_path = os.path.join(self.nli_data_dir, f"train_result.json") if self.nli_data_dir is not None else None
        self.valid_nli_data_path = os.path.join(self.nli_data_dir, f"test_result.json") if self.nli_data_dir is not None else None
        self.test_nli_data_path = None


def main():
    parser = TrlParser((SFTConfig, ModelConfig, DataArguments))
    training_args, model_config, data_config = parser.parse_args_and_config()
    is_main_process = training_args.local_rank in [-1, 0]

    # set seed
    set_seed(training_args.seed)

    # Prepare Model & Tokenizer
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # load base model from hf hub
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)
    tokenizer.padding_side = "right"

    # Tokenizer Configs
    if "llama" in model.config.model_type:
        # llama-3
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.eos_token = "<|eot_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.chat_template = "{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    elif "qwen2" in model.config.model_type:
        # qwen2
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|im_end|>"
        response_template = "<|im_start|>assistant\n"
        tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    elif "mistral" in model.config.model_type:
        # mistral-v0.1
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.eos_token = "</s>"
        response_template = "<s>assistant\n"
        tokenizer.chat_template = "{% for message in messages %}{{ '<s>' + message['role'] + '\n' + message['content'] + '</s>' }}{% endfor %}{% if add_generation_prompt %}{{ '<s>assistant\n' }}{% endif %}"

    assert tokenizer.pad_token != tokenizer.eos_token, "The pad_token_id and eos_token_id values of this tokenizer are identical. If you are planning for multi-turn training, it can result in the model continuously generating questions and answers without eos token. To avoid this, set the pad_token_id to a different value. (trl.trainer.utils.py, 129 line)"

    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Prepare Dataset
    train_dataset = data_config.preprocess_fct(
        data_config.train_data_path,
        data_config.max_history,
        tokenizer,
        eval=False,
        nli_data_path=data_config.train_nli_data_path,
        is_main_process=is_main_process,
    )
    valid_dataset = data_config.preprocess_fct(
        data_config.valid_data_path,
        data_config.max_history,
        tokenizer,
        eval=False,
        nli_data_path=data_config.valid_nli_data_path,
        is_main_process=is_main_process,
    )

    # Training
    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        peft_config=peft_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
