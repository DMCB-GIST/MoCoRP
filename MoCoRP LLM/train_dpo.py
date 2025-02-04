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
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from peft import PeftConfig, PeftModel

from utils.utils_dpo import build_dpo_dataset


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
        self.preprocess_fct = build_dpo_dataset
        self.train_data_path = os.path.join(self.dialog_data_dir, f"train_self_{'original' if self.original else 'revised'}.txt")
        self.valid_data_path = os.path.join(self.dialog_data_dir, f"valid_self_{'original' if self.original else 'revised'}.txt")
        self.test_data_path = os.path.join(self.dialog_data_dir, f"valid_self_{'original' if self.original else 'revised'}.txt")
        self.train_nli_data_path = os.path.join(self.nli_data_dir, f"train_result.json") if self.nli_data_dir is not None else None
        self.valid_nli_data_path = os.path.join(self.nli_data_dir, f"test_result.json") if self.nli_data_dir is not None else None
        self.test_nli_data_path = None


def main():
    parser = TrlParser((DPOConfig, ModelConfig, DataArguments))
    training_args, model_config, data_config = parser.parse_args_into_dataclasses()
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

    if not os.path.exists(os.path.join(model_config.model_name_or_path, "adapter_config.json")):
        # load base model from hf hub
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        # load PeftModel and make it to base model
        sft_peft_config = PeftConfig.from_pretrained(model_config.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(sft_peft_config.base_model_name_or_path, **model_kwargs)
        model = PeftModel.from_pretrained(model, model_config.model_name_or_path)
        model = model.merge_and_unload()

    peft_config = get_peft_config(model_config)
    if peft_config is None:
        # we have to load reference model for DPO
        ref_model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        # we don't have to load reference model if we train through LoRA
        ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)
    tokenizer.padding_side = "right"

    assert tokenizer.pad_token != tokenizer.eos_token, "The pad_token_id and eos_token_id values of this tokenizer are identical. If you are planning for multi-turn training, it can result in the model continuously generating questions and answers without eos token. To avoid this, set the pad_token_id to a different value. (trl.trainer.utils.py, 129 line)"

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
    trainer = DPOTrainer(
        model,
        ref_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
