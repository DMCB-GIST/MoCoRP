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


def build_sft_dataset(data_path, max_history, tokenizer, eval=False, nli_data_path=None, is_main_process=True):
    persona, history, candidate = get_dataset(data_path, nli_data_path)

    sft_dataset_dict = defaultdict(list)
    for dialog_idx in tqdm(range(len(persona)), disable=not is_main_process):
        if dialog_idx == 10: break
        current_persona = persona[dialog_idx]
        current_history = history[dialog_idx][-(2 * max_history + 1) :]
        current_response = candidate[dialog_idx][-1]

        if nli_data_path is None:
            # First stage Prompt
            # prompt = """Consider the context and the provided personas to generate a response.
            # Make sure to incorporate relevant details from the personas in your response.
            # Your persona: """ + "".join(current_persona)
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
            # prompt = """Consider the context and the provided personas to generate a response.
            # Refer to the information provided with each persona to determine its relevance to the current response.
            # Make sure to incorporate relevant details from the personas in your response.
            # Your persona: """ + "".join(current_persona)
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

        conversation = [{"role": "system", "content": prompt}]

        for i, h in enumerate(current_history):
            if i % 2 == 0:
                conversation.append({"role": "user", "content": h})
            else:
                conversation.append({"role": "assistant", "content": h})

        if not eval:
            # Training stage
            conversation.append({"role": "assistant", "content": current_response})

            input_ids = tokenizer.apply_chat_template(conversation, tokenize=True)
            attention_mask = [1 for _ in input_ids]
        else:
            # Evaluation stage
            input_ids = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
            attention_mask = [1 for _ in input_ids]

            prompt_len = len(input_ids)
            sft_dataset_dict["prompt_len"].append(prompt_len)

        sft_dataset_dict["input_ids"].append(input_ids)
        sft_dataset_dict["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(sft_dataset_dict)

    return dataset
