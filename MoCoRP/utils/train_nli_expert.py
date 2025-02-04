import os
import re

import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .utils_nli_expert import MODEL_INPUT


def train(args, train_dataset, valid_dataset, model, tokenizer, logger):
    train_loader = DataLoader(train_dataset, batch_size=args.nli_batch_size, shuffle=True)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_training_steps = args.num_train_epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num examples = {len(train_dataset):,}")
    logger.info(f"  Total optimization steps = {num_training_steps:,}")
    logger.info(f"  Instantaneous batch size per GPU = {args.nli_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.nli_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))

    # Begin training
    global_step = 0
    for epoch in range(1, args.num_train_epochs + 1):
        # Apply Early stopping
        if args.early_stop_epochs > 1 and args.early_stop_epochs == epoch:
            break

        # Train with Dialogue NLI
        model.train()
        mc_loss = []
        accuracy = []
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            batch = {k: input_tensor.to(args.device) for k, input_tensor in zip(MODEL_INPUT, batch)}

            result = model(**batch)
            pred = result.logits.argmax(-1)
            mc_loss.append(result.loss.item())
            accuracy.extend((pred == batch["labels"].reshape(-1)).cpu().tolist())
            loss = result.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            global_step += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            pbar.set_postfix(mc_loss=np.mean(mc_loss), accuracy=np.mean(accuracy))

        # Evaluate
        evaluate(args, valid_dataset, model, tokenizer, logger)

        # Save model
        output_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Complete training Epoch: {epoch}, global_step: {global_step:,}")


def evaluate(args, eval_dataset, model, tokenizer, logger):
    eval_loader = DataLoader(eval_dataset, batch_size=args.nli_batch_size, shuffle=False)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset):,}")
    logger.info(f"  Batch size = {args.nli_batch_size}")

    model.eval()
    mc_loss = []
    accuracy = []
    logits = []
    with torch.no_grad():
        pbar = tqdm(eval_loader)
        for step, batch in enumerate(pbar):
            batch = {k: input_tensor.to(args.device) for k, input_tensor in zip(MODEL_INPUT, batch)}

            result = model(**batch)
            pred = result.logits.argmax(-1)
            mc_loss.append(result.loss.item())
            accuracy.extend((pred == batch["labels"].reshape(-1)).cpu().tolist())
            logits.extend(result.logits.cpu().tolist())

            pbar.set_postfix(mc_loss=np.mean(mc_loss), accuracy=np.mean(accuracy))

    with open("data/dialogue_nli/dialogue_nli_train.jsonl") as fp:
        data = json.load(fp)
    data = [{**d, "logits": logit} for d, logit in zip(data, logits)]
    with open("data/dialogue_nli/dialogue_nli_train.jsonl", "w") as fp:
        json.dump(data, fp)
    result = {
        "mc_loss": np.mean(mc_loss) * 100,
        "accuracy": np.mean(accuracy) * 100,
    }

    for k, v in result.items():
        result[k] = round(v, 2)

    logger.info("Complete Evaulation")
    logger.info(f"Eval result: {result}")
