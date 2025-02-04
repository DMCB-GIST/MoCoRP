import os
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from tqdm.auto import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.data.data_collator import DataCollatorWithPadding
from trl import (
    SFTConfig,
    SFTTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
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

    max_new_tokens: Optional[int] = field(default=50)
    generate_train: Optional[bool] = field(default=False, metadata={"help": ""})

    def __post_init__(self):
        self.preprocess_fct = build_sft_dataset
        self.train_data_path = os.path.join(self.dialog_data_dir, f"train_self_{'original' if self.original else 'revised'}.txt")
        self.valid_data_path = os.path.join(self.dialog_data_dir, f"valid_self_{'original' if self.original else 'revised'}.txt")
        self.test_data_path = os.path.join(self.dialog_data_dir, f"valid_self_{'original' if self.original else 'revised'}.txt")
        self.train_nli_data_path = os.path.join(self.nli_data_dir, f"train_result.json") if self.nli_data_dir is not None else None
        self.valid_nli_data_path = os.path.join(self.nli_data_dir, f"test_result.json") if self.nli_data_dir is not None else None
        self.test_nli_data_path = None

        # two stage training을 위한 train dataset 생성
        if self.generate_train:
            self.test_data_path = self.train_data_path


def main():
    parser = TrlParser((SFTConfig, ModelConfig, DataArguments))
    training_args, model_config, data_config = parser.parse_args_and_config()

    # Setup Distributed setting
    mixed_precision = "fp16" if training_args.fp16 else "bf16" if training_args.bf16 else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)

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

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)

    # 모델을 각 GPU에 배치함
    model = accelerator.prepare_model(model, evaluation_mode=True)
    unwrapped_model = accelerator.unwrap_model(model)

    # Prepare Dataset
    test_dataset = data_config.preprocess_fct(
        data_config.test_data_path,
        data_config.max_history,
        tokenizer,
        eval=True,
        nli_data_path=data_config.test_nli_data_path,
        is_main_process=accelerator.is_main_process,
    )

    # Setup Data Loader
    test_dataset = test_dataset.with_format("torch")

    # Reference: Trainer.get_test_dataloader (line 995)
    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader_params = {
        "batch_size": training_args.per_device_eval_batch_size,
        "collate_fn": data_collator,
        "num_workers": training_args.dataloader_num_workers,
        "pin_memory": training_args.dataloader_pin_memory,
        "persistent_workers": training_args.dataloader_persistent_workers,
        "sampler": None,
        "drop_last": training_args.dataloader_drop_last,
        "prefetch_factor": training_args.dataloader_prefetch_factor,
    }

    test_dataloader = DataLoader(test_dataset, **dataloader_params)

    # 데이터 로더를 각 GPU에 배치함
    test_dataloader = accelerator.prepare(test_dataloader)

    # Generate output
    output_dict = {"predictions": []}
    for step, batch in enumerate(tqdm(test_dataloader, disable=not accelerator.is_main_process)):
        prompt_len = batch.pop("prompt_len")

        with accelerator.autocast():
            result = unwrapped_model.generate(
                **batch,
                max_new_tokens=data_config.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 프롬프트 제거
        result = result[:, prompt_len:]

        # Distributed setting에서 텐서를 gather 하기 위해 패딩
        padded_logits = accelerator.pad_across_processes(result, dim=1, pad_index=tokenizer.pad_token_id)

        # 여러 GPU에 흩뿌려져 있던 텐서를 gather
        gathered_logits = accelerator.gather_for_metrics(padded_logits)

        # 후처리 된 predictions
        predictions = [
            pred.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").strip()
            for pred in tokenizer.batch_decode(gathered_logits)
        ]
        output_dict["predictions"].extend(predictions)

    # Save results and this is only executed in the main process!
    if accelerator.is_main_process:
        file_path = "test_result.json" if not data_config.generate_train else "train_result.json"
        save_path = os.path.join(training_args.output_dir, file_path)
        with open(save_path, "w") as fp:
            json.dump(output_dict, fp)


if __name__ == "__main__":
    main()
