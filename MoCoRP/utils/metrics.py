import os
import re
import json
from collections import Counter

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from nltk.translate import bleu_score as nltkbleu
import rouge
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .data_utils import pad_input

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess, answers):
    """
    Return the max F1 score between the guess and *any* answer.
    """
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


def _bleu(guess, answers, weights=None):
    """
    Compute approximate BLEU score between guess and a set of answers.
    """
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    if weights is None:
        # default bleu-4
        weights = [1 / 4 for _ in range(4)]
    return nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
        weights=weights,
    )


def _rouge(guess, answers):
    global rouge
    """Compute ROUGE score between guess and *any* answers. Return the best."""
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    scores = [
        evaluator.get_scores(normalize_answer(guess), normalize_answer(a))
        for a in answers
    ]

    scores_rouge1 = [score['rouge-1']['r'] for score in scores]
    scores_rouge2 = [score['rouge-2']['r'] for score in scores]
    scores_rougeL = [score['rouge-l']['r'] for score in scores]
    return max(scores_rouge1), max(scores_rouge2), max(scores_rougeL)


def get_parlai_metric(prediction, response):
    """
    * This function is an evaluation function for ConvAI2 and was implemented using the official ParlAI library, producing identical results.
    Computes evaluation metrics for dialogue generation, including F1, BLEU (1-4), and ROUGE (1, 2, L).
    
    Args:
        prediction (list of str): A list of generated responses.
        response (list of str): A list of ground truth responses.
        
    Returns:
        dict: A dictionary containing the following evaluation metrics (scaled to percentage):
            - "F1": Average token-level F1 score.
            - "BLEU-1": BLEU score for 1-gram precision.
            - "BLEU-2": BLEU score for 2-gram precision.
            - "BLEU-3": BLEU score for 3-gram precision.
            - "BLEU-4": BLEU score for 4-gram precision.
            - "ROUGE-1": ROUGE-1 score (unigram overlap).
            - "ROUGE-2": ROUGE-2 score (bigram overlap).
            - "ROUGE-L": ROUGE-L score (longest common subsequence).
    """

    result = dict()

    f1 = [_f1_score(pred, (resp, )) for pred, resp in zip(prediction, response)]
    result["F1"] = np.mean(f1) * 100

    bleu1 = [_bleu(pred, (resp, ), [1 / (1) for _ in range(1)]) for pred, resp in zip(prediction, response)]
    bleu2 = [_bleu(pred, (resp, ), [1 / (2) for _ in range(2)]) for pred, resp in zip(prediction, response)]
    bleu3 = [_bleu(pred, (resp, ), [1 / (3) for _ in range(3)]) for pred, resp in zip(prediction, response)]
    bleu4 = [_bleu(pred, (resp, )) for pred, resp in zip(prediction, response)]
    result["BLEU-1"] = np.mean(bleu1) * 100
    result["BLEU-2"] = np.mean(bleu2) * 100
    result["BLEU-3"] = np.mean(bleu3) * 100
    result["BLEU-4"] = np.mean(bleu4) * 100

    rouge = [_rouge(pred, (resp, )) for pred, resp in zip(prediction, response)]
    rouge1, rouge2, rougeL = [r1 for r1, _, _ in rouge], [r2 for _, r2, _ in rouge], [rL for _, _, rL in rouge]
    result["ROUGE-1"] = np.mean(rouge1) * 100
    result["ROUGE-2"] = np.mean(rouge2) * 100
    result["ROUGE-L"] = np.mean(rougeL) * 100

    return result


def compute_metrics_from_logits(data_args, logits, targets):
    """
        Hits@k or Recall@k for N candidates, and MRR (if task is mpchat)

        logits: (batch_size, num_candidates)
        targets: (batch_size, )
    """
    batch_size, num_candidates = logits.shape

    sorted_indices = logits.sort(descending=True)[1]
    targets = targets.tolist()

    # hits@k or recall@k
    result = dict()
    metric_name = "Hits" if data_args.task == "convai2" else "R"
    ks = [1, 2]
    # ks = [1, 2, 5, 10] if data_args.task == "convai2" else [1, 5, 10, 50]
    for k in ks:
        # sorted_indices[:, :k]: (batch_size, k)
        num_ok = 0
        for tgt, topk in zip(targets, sorted_indices[:, :k].tolist()):
            if tgt in topk:
                num_ok += 1
        result[f"{metric_name}@{k}"] = num_ok / batch_size * 100

    if data_args.task == "mpchat":
        # MRR
        MRR = 0
        for tgt, topk in zip(targets, sorted_indices.tolist()):
            rank = topk.index(tgt) + 1
            MRR += 1 / rank
        MRR = MRR / batch_size * 100
        result["MRR"] = MRR

    return result


def compute_c_score_from_predictions(model_args, data_args, training_args, predictions, logger):
    # get personas
    if data_args.task == "convai2":
        with open(data_args.test_data_path) as fp:
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

        personas = []
        for lines in convai:
            current_persona = []

            lines = [line.strip() for line in lines]
            for line in lines:
                line = re.sub(r"^\d+\s", "", line)  # remove numeric prefix

                if "your persona: " in line:  # persona
                    line = line.replace("your persona: ", "")
                    current_persona.append(line)
                else:  # utterance preprocess
                    personas.append(current_persona)
    elif data_args.task == "mpchat":
        with open(data_args.test_data_path) as fp:
            data = json.load(fp)

        personas = []
        for dialog_idx, dialog in enumerate(data):
            main_author = dialog["main_author"]
            turn_indices = []
            for turn_idx, author in enumerate(dialog["authors"]):
                if main_author == author:
                    turn_indices.append(turn_idx)

            for turn_idx in turn_indices:
                personas.append([x["title"] for x in dialog["candidate_personas"]])

    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.nli_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.nli_model_name_or_path)
    model.to(training_args.device)

    # build persona, prediction pair dataset
    dataset = {"input_ids": [], "attention_mask": []}

    for persona, prediction in zip(personas, predictions):
        for p in persona:
            input_sent = tokenizer.cls_token + p + tokenizer.sep_token + prediction
            tokenized_input = tokenizer(input_sent, add_special_tokens=False)

            input_ids = tokenized_input["input_ids"]
            attention_mask = tokenized_input["attention_mask"]

            dataset["input_ids"].append(input_ids)
            dataset["attention_mask"].append(attention_mask)

    nli_dataset = pad_input(dataset, tokenizer=tokenizer, num_candidates=1)
    nli_dataset = TensorDataset(*nli_dataset)
    nli_loader = DataLoader(nli_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)

    model.eval()
    # C score: 0 for neutral, +1 for entail, -1 for contradict
    label2score = {"neutral": 0, "positive": 1, "negative": -1}
    nli_result = []
    c_score = []
    logits = []

    with torch.no_grad():
        progress_bar = tqdm(nli_loader, desc="C score")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(model.device) for k, v in zip(dataset.keys(), batch)}

            result = model(**batch)
            pred = result.logits.argmax(-1).cpu().tolist()
            label = list(map(lambda x: model.config.id2label[x], pred))
            score = list(map(lambda x: label2score[x], label))
            logit = result.logits.cpu().tolist()

            nli_result.extend(label)
            c_score.extend(score)
            logits.extend(logit)
            progress_bar.set_postfix(c_score=np.mean(c_score) * 100)

    return nli_result, np.mean(c_score) * 100, logits
