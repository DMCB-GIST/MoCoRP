import os
import re
import argparse

import json
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.data_utils import pad_input


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dnli_data_dir", type=str, default="data/dialogue_nli")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--nli_data_path", type=str)
    parser.add_argument("--task", default="convai2", type=str)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nli_batch_size", default=64, type=int)

    args = parser.parse_args()

    # Build dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # Load dataset
    if args.task == "convai2":
        with open(args.data_path) as fp:
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

                    persona.append(current_persona)
                    history.append(current_history[:-1])
                    candidate.append(current_candidate)
        candidate = [cand[-1] for cand in candidate]
    elif args.task == "mpchat":

        def remove_special_characters(text):
            return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()

        with open(args.data_path) as fp:
            data = json.load(fp)

        # build MPChat
        persona = []
        history = []
        candidate = []
        for dialog_idx, dialog in enumerate(data):
            main_author = dialog["main_author"]
            turn_indices = []
            for turn_idx, author in enumerate(dialog["authors"]):
                if main_author == author:
                    turn_indices.append(turn_idx)

            for turn_idx in turn_indices:
                persona_sentences = [x["title"] for x in dialog["candidate_personas"]]
                dialog_history = [remove_special_characters(h) for h in dialog["messages"][:turn_idx]]
                response = dialog["messages"][turn_idx]

                persona.append(persona_sentences)
                history.append(dialog_history)
                candidate.append(response)

    # tokenize
    dataset = {"input_ids": [], "attention_mask": []}

    for persona_example, cand in tqdm(zip(persona, candidate), total=len(candidate)):
        for p in persona_example:
            input_sent = tokenizer.cls_token + p + tokenizer.sep_token + cand

            tokenized_input = tokenizer(input_sent, add_special_tokens=False, truncation=True)
            input_ids = tokenized_input["input_ids"]
            attention_mask = tokenized_input["attention_mask"]

            dataset["input_ids"].append(input_ids)
            dataset["attention_mask"].append(attention_mask)

    nli_dataset = pad_input(dataset, tokenizer=tokenizer, num_candidates=1)
    nli_dataset = TensorDataset(*nli_dataset)
    dataloader = DataLoader(nli_dataset, batch_size=args.nli_batch_size, shuffle=False, pin_memory=True)

    # C score: 0 for neutral, +1 for entail, -1 for contradict
    label2score = {"neutral": 0, "positive": 1, "negative": -1}

    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint)
    model.to(args.device)

    model.eval()
    c_label = []
    c_score = []
    logits = []
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for step, batch in enumerate(pbar):
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, attention_mask = batch

            result = model(input_ids=input_ids, attention_mask=attention_mask)

            pred = result.logits.argmax(-1).cpu().tolist()
            label = list(map(lambda x: model.config.id2label[x], pred))
            score = list(map(lambda x: label2score[x], label))
            logit = result.logits.cpu().tolist()

            c_label.extend(label)
            c_score.extend(score)
            logits.extend(logit)
            pbar.set_postfix(score=np.mean(c_score) * 100)

    average_c_score = np.mean(c_score) * 100

    with open(args.nli_data_path, "w") as fp:
        # (MoCoRP) computing data relations done
        json.dump({"nli_result": c_label, "nli_logits": logits}, fp)


if __name__ == "__main__":
    main()
