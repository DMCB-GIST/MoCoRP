import re
import os
from collections import defaultdict

from tqdm.auto import tqdm
import json
import numpy as np
import torch
from datasets import Dataset

entail, neutral, contradict = ["(Entailment)", "(Neutral)", "(Contradict)"]


def get_dataset(data_path, nli_data_path=None):
    # two stage training
    if nli_data_path is not None:
        with open(nli_data_path) as fp:
            nli2tag = {"positive": entail, "neutral": neutral, "negative": contradict}
            nli_tag = (nli2tag[nli] for nli in json.load(fp)["nli_result"])

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

                if nli_data_path is not None:
                    nli_persona = [p + next(nli_tag) for p in current_persona]
                    persona.append(nli_persona)
                else:
                    persona.append(current_persona)
                history.append(current_history[:-1])
                candidate.append(current_candidate)

    return persona, history, candidate


def build_dpo_dataset(data_path, max_history, tokenizer, eval=False, nli_data_path=None, is_main_process=True):
    persona, history, candidate = get_dataset(data_path, nli_data_path)

    dpo_dataset_dict = defaultdict(list)
    for dialog_idx in tqdm(range(len(persona)), disable=not is_main_process):
        # if dialog_idx == 10: break
        current_persona = persona[dialog_idx]
        current_history = history[dialog_idx][-(2 * max_history + 1) :]
        current_response = candidate[dialog_idx][-1]
        current_distractor = candidate[dialog_idx][-2]

        if nli_data_path is None:
            # First stage Prompt
            prompt = """Review the previous conversation context and your personas.
Think about what aspect of your persona is most relevant to respond to the user's statement.
Break down your reasoning process step by step as follows:

Step 1: Identify the main point of the user's statement.
Step 2: Recall relevant persona traits that align with this point.
Step 3: Synthesize a response that highlights those traits while keeping the conversation engaging and natural.
Step 4: Generate the final response.

Your persona:
{persona}

Generate your final response based on these steps.""".format(persona="".join([f"- {p}\n" for p in current_persona]))
        else:
            # Second stage Prompt
            prompt = """Review the previous conversation context and your personas provided with labels in parentheses.
Think about what aspect of your persona is most relevant to respond to the user's statement.

Each label is associated with a persona, indicating that it has a relation to the expected final response.
- Entail: The information in the persona includes or entails the response.
For example, the persona is 'I love animals' and the response is 'I love cats'
- Contradict: The persona and response contradict each other.
For example, the persona is 'I love animals' and the response is 'I hate cats'
- Neutral: The persona and response are neither in direct contradiction nor entailment.
For example, the persona is 'I love animals' and the response is 'Cats are lovable'

Break down your reasoning process step by step as follows:

Step 1: Identify the main point of the user's statement.
Step 2: Recall relevant persona traits that align with this point.
Step 3: Refer to the label provided for each persona to determine whether or not to include that information in the response.
Step 4: Synthesize a response that highlights those traits while keeping the conversation engaging and natural.
Step 5: Generate the final response.

Your persona:
{persona}

Generate your final response based on these steps.""".format(persona="".join([f"- {p}\n" for p in current_persona]))

        # DPOTrainer가 prompt_input_ids를 요구함
        prompt_conv = [{"role": "system", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(prompt_conv, tokenize=False)

        conversation = []
        for i, h in enumerate(current_history):
            if i % 2 == 0:
                conversation.append({"role": "user", "content": h})
            else:
                conversation.append({"role": "assistant", "content": h})

        if not eval:
            # Training stage
            chosen_conv = conversation + [{"role": "assistant", "content": current_response}]
            rejected_conv = conversation + [{"role": "assistant", "content": current_distractor}]

            chat_chosen = tokenizer.apply_chat_template(chosen_conv, tokenize=False)
            chat_rejected = tokenizer.apply_chat_template(rejected_conv, tokenize=False)

            # prompt + history를 일치시켜줌, system_instruction과 history를 각각 인코딩하면 달라지는데 이를 확인
            assert tokenizer.apply_chat_template(prompt_conv + chosen_conv, tokenize=False) == chat_prompt + chat_chosen
            assert tokenizer.apply_chat_template(prompt_conv + rejected_conv, tokenize=False) == chat_prompt + chat_rejected

            # 실제 chosen/rejected input은 prompt + chosen/rejected로 설정되므로 append 시에는 chat_chosen/rejected로 해야 함
            dpo_dataset_dict["prompt"].append(chat_prompt)
            dpo_dataset_dict["chosen"].append(chat_chosen)
            dpo_dataset_dict["rejected"].append(chat_rejected)
        else:
            # Evaluation stage
            chat_history = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

            # 동일하게 prompt + history를 일치시켜줌
            assert tokenizer.apply_chat_template(prompt_conv + conversation, tokenize=False, add_generation_prompt=True) == chat_prompt + chat_history

            tokenized = tokenizer(chat_prompt + chat_history, add_special_tokens=False)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            dpo_dataset_dict["input_ids"].append(input_ids)
            dpo_dataset_dict["attention_mask"].append(attention_mask)

            prompt_len = len(input_ids)
            dpo_dataset_dict["prompt_len"].append(prompt_len)

    dataset = Dataset.from_dict(dpo_dataset_dict)

    return dataset
