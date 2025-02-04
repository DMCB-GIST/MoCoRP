import re
import os
import random
from collections import defaultdict

from tqdm.auto import tqdm
import json
import jsonlines
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

human, bot, resp = ["<human>", "<bot>", "<resp>"]


class DialogDataset(Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        labels,
        mc_token_ids,
        mc_labels,
        rp_token_ids,
        rp_labels=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.decoder_attention_mask = decoder_attention_mask
        self.labels = labels
        self.mc_token_ids = mc_token_ids
        self.mc_labels = mc_labels
        self.rp_token_ids = rp_token_ids
        self.rp_labels = rp_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        feature = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "decoder_input_ids": self.decoder_input_ids[idx],
            "decoder_attention_mask": self.decoder_attention_mask[idx],
            "labels": self.labels[idx],
            "mc_token_ids": self.mc_token_ids[idx],
            "mc_labels": self.mc_labels[idx],
            "rp_token_ids": self.rp_token_ids[idx],
        }

        if self.rp_labels is not None:
            feature["rp_labels"] = self.rp_labels[idx]

        return feature


def build_convai(data_path, max_history, num_candidates, tokenizer, nli_data_path=None):
    # NLI tags
    if nli_data_path is not None:
        with open(nli_data_path) as fp:
            nli_tag = (nli for nli in json.load(fp)["nli_logits"])

    with open(data_path) as fp:
        convai_dataset = fp.readlines()

    convai = []
    current_sentence = []

    for line in convai_dataset:
        if line.startswith("1 ") and current_sentence:  # append current dialogue
            convai.append(current_sentence)
            current_sentence = [line]
        else:
            current_sentence.append(line)
    convai.append(current_sentence)  # last sentence

    persona = []
    history = []
    candidate = []
    rp_label = []
    for lines in convai:
        current_persona = []
        current_history = []

        lines = [line.strip() for line in lines]
        for line in lines:
            line = re.sub(r"^\d+\s", "", line)  # remove numeric prefix

            if "your persona: " in line:  # persona
                line = line.replace("your persona: ", "")
                current_persona.append(line)
            else:  # utterance preprocess
                utterances, current_candidate = line.split("\t\t")
                human_utterance, bot_utterance = utterances.split("\t")
                current_candidate = current_candidate.split("|")

                current_history.extend([human_utterance, bot_utterance])

                persona.append([tokenizer.mask_token + p for p in current_persona])
                if nli_data_path is not None:
                    rp_label.append([next(nli_tag) for _ in range(len(current_persona))])

                history.append(current_history[:-1])
                candidate.append(current_candidate)

    dataset = defaultdict(list)

    progress_bar = tqdm(range(len(persona)), desc="Fetching ConvAI2")
    for dialog_idx in progress_bar:
        context = "".join(
            persona[dialog_idx]
            + [bot + h if i % 2 else human + h for i, h in enumerate(history[dialog_idx][-(2 * max_history + 1) :])]
            + [tokenizer.eos_token]
        )
        context_inputs = tokenizer(context, add_special_tokens=False)
        input_ids = context_inputs["input_ids"]
        attention_mask = context_inputs["attention_mask"]

        rp_token_ids = [
            idx for idx, token in enumerate(input_ids)
            if token in [tokenizer.mask_token_id]
        ]
        if nli_data_path is not None:
            rp_labels = rp_label[dialog_idx]

        # n-1 for distractor candidates, 1 for gold response
        for cand_count, cand in enumerate(candidate[dialog_idx][-num_candidates:][::-1]):
            response = resp + cand + tokenizer.eos_token
            response_inputs = tokenizer(response, add_special_tokens=False)
            decoder_input_ids = response_inputs["input_ids"][:-1]
            decoder_attention_mask = response_inputs["attention_mask"][:-1]
            mc_token_ids = [len(decoder_input_ids) - 1]

            if cand_count == 0:
                # gold response
                labels = response_inputs["input_ids"][1:]
                mc_labels = [1.0]
            else:
                # distractor response
                labels = [-100] * len(response_inputs["input_ids"][1:])
                mc_labels = [0.0]

            dataset["input_ids"].append(input_ids)
            dataset["attention_mask"].append(attention_mask)
            dataset["decoder_input_ids"].append(decoder_input_ids)
            dataset["decoder_attention_mask"].append(decoder_attention_mask)
            dataset["labels"].append(labels)
            dataset["mc_token_ids"].append(mc_token_ids)
            dataset["mc_labels"].append(mc_labels)
            dataset["rp_token_ids"].append(rp_token_ids)
            if nli_data_path is not None:
                dataset["rp_labels"].append(rp_labels)

    tensor_dataset = pad_input(dataset, tokenizer=tokenizer, num_candidates=num_candidates)
    tensor_dataset = DialogDataset(*tensor_dataset)

    return tensor_dataset


def remove_special_characters(text):
    # remove special characters in utterance
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()


def build_mpchat(data_path, max_history, num_candidates, tokenizer, nli_data_path=None):
    # NLI tags
    if nli_data_path is not None:
        with open(nli_data_path) as fp:
            nli_tag = (nli for nli in json.load(fp)["nli_logits"])

    with open(data_path) as fp:
        data = json.load(fp)

    dataset = defaultdict(list)

    # create candidate pool
    if "train" in data_path:
        candidates = []
        for dialog_idx, dialog in enumerate(data):
            main_author = dialog["main_author"]
            for turn_idx, author in enumerate(dialog["authors"]):
                if main_author == author:
                    candidates.append((dialog_idx, dialog["messages"][turn_idx]))

    # build MPChat
    progress_bar = tqdm(data, desc="Fetching MPChat")
    for dialog_idx, dialog in enumerate(progress_bar):
        # if dialog_idx == 300: break
        main_author = dialog["main_author"]
        turn_indices = []
        for turn_idx, author in enumerate(dialog["authors"]):
            if main_author == author:
                turn_indices.append(turn_idx)

        for turn_idx in turn_indices:
            persona_sentences = [x["title"] for x in dialog["candidate_personas"]]
            persona_sentences = [tokenizer.mask_token + p for p in persona_sentences]

            if nli_data_path is not None:
                rp_labels = [next(nli_tag) for _ in range(len(persona_sentences))]

            dialog_history = [
                [remove_special_characters(h), author]
                for h, author in zip(
                    dialog["messages"][:turn_idx][-(2 * max_history + 1) :],
                    dialog["authors"][:turn_idx][-(2 * max_history + 1) :],
                )
            ]

            if "train" in data_path:
                # in train mode, response and distractors consist the number of num_candidates
                candidate_example = []
                for cand_count in range(num_candidates):
                    if cand_count == 0:
                        # gold response
                        cand = dialog["messages"][turn_idx]
                        candidate_example.append(cand)
                    else:
                        # distractor response
                        while True:
                            cand_dialog_idx, cand = random.choice(candidates)
                            if cand_dialog_idx != dialog_idx:
                                candidate_example.append(cand)
                                break
            else:
                # in val, test mode, response and distractors consist 100 of candidates
                candidate_example = dialog["nrp_candidate_responses"][turn_idx]

            context = "".join(
                persona_sentences
                + [bot + h if author == main_author else human + h for h, author in dialog_history]
                + [tokenizer.eos_token]
            )
            context_inputs = tokenizer(context, add_special_tokens=False)
            input_ids = context_inputs["input_ids"]
            attention_mask = context_inputs["attention_mask"]

            rp_token_ids = [
                idx for idx, token in enumerate(input_ids)
                if token in [tokenizer.mask_token_id]
            ]

            for cand_count, cand in enumerate(candidate_example):
                response = resp + cand + tokenizer.eos_token
                response_inputs = tokenizer(response, add_special_tokens=False)
                decoder_input_ids = response_inputs["input_ids"][:-1]
                decoder_attention_mask = response_inputs["attention_mask"][:-1]
                mc_token_ids = [len(decoder_input_ids) - 1]

                if cand_count == 0:
                    # gold response
                    labels = response_inputs["input_ids"][1:]
                    mc_labels = [1.0]
                else:
                    # distractor response
                    labels = [-100] * len(response_inputs["input_ids"][1:])
                    mc_labels = [0.0]

                dataset["input_ids"].append(input_ids)
                dataset["attention_mask"].append(attention_mask)
                dataset["decoder_input_ids"].append(decoder_input_ids)
                dataset["decoder_attention_mask"].append(decoder_attention_mask)
                dataset["labels"].append(labels)
                dataset["mc_token_ids"].append(mc_token_ids)
                dataset["mc_labels"].append(mc_labels)
                dataset["rp_token_ids"].append(rp_token_ids)
                if nli_data_path is not None:
                    dataset["rp_labels"].append(rp_labels)

    tensor_dataset = pad_input(dataset, tokenizer=tokenizer, num_candidates=num_candidates)
    tensor_dataset = DialogDataset(*tensor_dataset)

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
        if "rp_labels" == key:
            tensor = tensor.reshape(-1, num_candidates, tensor.size(-2), tensor.size(-1))
        elif num_candidates != 1:
            tensor = tensor.reshape(-1, num_candidates, tensor.size(-1))
        tensor_dataset.append(tensor)
    return tensor_dataset


def build_dnli(dnli_data_path, task, num_candidates, tokenizer):
    with jsonlines.open(dnli_data_path) as fp:
        data = fp.read()
        if task == "convai2":
            data = data[:13000]
        else:
            data = data[:1100]

    dataset = defaultdict(list)

    progress_bar = tqdm(data, desc="Fetching DialogueNLI")
    for dialog_idx, dialog in enumerate(progress_bar):
        encoder_input = tokenizer.mask_token + dialog["sentence1"] + tokenizer.eos_token
        tokenized_encoder_input = tokenizer(encoder_input, add_special_tokens=False)
        input_ids = tokenized_encoder_input["input_ids"]
        attention_mask = tokenized_encoder_input["attention_mask"]

        decoder_input = resp + dialog["sentence2"] + tokenizer.eos_token
        tokenized_decoder_input = tokenizer(decoder_input, add_special_tokens=False)
        decoder_input_ids = tokenized_decoder_input["input_ids"][:-1]
        decoder_attention_mask = tokenized_decoder_input["attention_mask"][:-1]

        mc_token_ids = [len(decoder_input_ids) - 1]
        rp_token_ids = [0]
        if dialog["label"] == "neutral":
            labels = [-100] * len(tokenized_decoder_input["input_ids"][1:])
            mc_labels = [0.0]
            if dialog["logits"].index(max(dialog["logits"])) == 0:
                rp_labels = [dialog["logits"]]
            else:
                rp_labels = [[1.0, 0.0, 0.0]]
        elif dialog["label"] == "positive":
            if task == "convai2":
                labels = [-100] * len(tokenized_decoder_input["input_ids"][1:]) # ConvAI2
            else:
                labels = tokenized_decoder_input["input_ids"][1:] # MPChat
            mc_labels = [1.0]
            if dialog["logits"].index(max(dialog["logits"])) == 1:
                rp_labels = [dialog["logits"]]
            else:
                rp_labels = [[0.0, 1.0, 0.0]]
        else:
            labels = [-100] * len(tokenized_decoder_input["input_ids"][1:])
            mc_labels = [0.0]
            if dialog["logits"].index(max(dialog["logits"])) == 2:
                rp_labels = [dialog["logits"]]
            else:
                rp_labels = [[0.0, 0.0, 1.0]]

        dataset["input_ids"].append(input_ids)
        dataset["attention_mask"].append(attention_mask)
        dataset["decoder_input_ids"].append(decoder_input_ids)
        dataset["decoder_attention_mask"].append(decoder_attention_mask)
        dataset["labels"].append(labels)
        dataset["mc_token_ids"].append(mc_token_ids)
        dataset["mc_labels"].append(mc_labels)
        dataset["rp_token_ids"].append(rp_token_ids)
        dataset["rp_labels"].append(rp_labels)

    tensor_dataset = pad_input(dataset, tokenizer=tokenizer, num_candidates=num_candidates)
    tensor_dataset = DialogDataset(*tensor_dataset)

    return tensor_dataset
