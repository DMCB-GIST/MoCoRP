import os

import jsonlines
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence


MODEL_INPUT = [
    "input_ids", "attention_mask", "labels"
]


def build_dnli(dataset_path, tokenizer, mode):
    with jsonlines.open(os.path.join(dataset_path, f"dialogue_nli_{mode}.jsonl")) as f:
        data = f.read()

    dataset = {k: [] for k in MODEL_INPUT}

    for dialog_idx, dialog in enumerate(data):
        input_sent = tokenizer.cls_token + dialog["sentence1"] + tokenizer.sep_token + dialog["sentence2"]
        tokenized_input = tokenizer(input_sent, add_special_tokens=False)

        input_ids = tokenized_input["input_ids"]
        attention_mask = tokenized_input["attention_mask"]
        if dialog["label"] == "neutral":
            labels = [0]
        elif dialog["label"] == "positive":
            labels = [1]
        else:
            labels = [2]

        dataset["input_ids"].append(input_ids)
        dataset["attention_mask"].append(attention_mask)
        dataset["labels"].append(labels)

    tensor_dataset = pad_input(dataset, tokenizer=tokenizer, num_candidates=1)
    tensor_dataset = TensorDataset(*tensor_dataset)

    return tensor_dataset


def pad_input(dataset, tokenizer, num_candidates):
    tensor_dataset = []
    for key, value in dataset.items():
        padding_value = 0
        if "input_ids" in key:
            padding_value = tokenizer.pad_token_id
        elif "labels" in key:
            padding_value = -100

        tensor = pad_sequence(
            [torch.from_numpy(np.array(x)) for x in value], batch_first=True, padding_value=padding_value
        )
        if num_candidates != 1:
            tensor = tensor.reshape(-1, num_candidates, tensor.size(-1))
        tensor_dataset.append(tensor)
    return tensor_dataset
