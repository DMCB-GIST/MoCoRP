import os

import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .metrics import get_parlai_metric, compute_metrics_from_logits, compute_c_score_from_predictions


def train(model_args, data_args, training_args, nli_dataset, train_dataset, valid_dataset, model, tokenizer, logger):
    logger.info("Prepare training Data Loader")
    nli_dataloader = DataLoader(nli_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)

    logger.info("Setup AdamW optimizer and LR scheduler")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    num_training_steps = training_args.num_train_epochs * (len(train_dataloader) + len(nli_dataloader))
    num_warmup_steps = int(num_training_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    logger.info("***** Running training *****")
    global_step = 0
    num_train_epochs = int(
        training_args.num_train_epochs
        if model_args.early_stop_epochs is None
        else min(training_args.num_train_epochs, model_args.early_stop_epochs)
    )

    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Num main task examples = {len(train_dataset):,}")
    logger.info(f"  Total optimization steps = {num_training_steps:,}")
    logger.info(f"  Instantaneous batch size per GPU = {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(
        "  Total main task train batch size (w. parallel, distributed & accumulation) = %d",
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if training_args.ddp_backend is not None else 1),
    )

    logger.info("Begin training!")
    for epoch in range(num_train_epochs):
        model.train()

        lm_loss = []
        mc_loss = []
        rp_loss = []
        logger.info(f"Epoch - {epoch + 1}: NLI dataset ({data_args.task})")
        logging_steps = int(len(nli_dataloader) * model_args.logging_ratio)
        progress_bar = tqdm(nli_dataloader, desc=f"Epoch - {epoch + 1}", leave=False)
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            result = model(**batch)
            if result.loss is not None:
                lm_loss.append(result.loss.cpu().detach().item())
            mc_loss.append(result.mc_loss.cpu().detach().item())
            rp_loss.append(result.rp_loss.cpu().detach().item())

            if result.loss is not None:
                loss = result.loss * model_args.lm_coef + result.mc_loss * model_args.mc_coef + result.rp_loss * model_args.rp_coef
            else:
                loss = result.mc_loss * model_args.mc_coef + result.rp_loss * model_args.rp_coef

            if training_args.gradient_accumulation_steps > 1:
                loss = loss / training_args.gradient_accumulation_steps

            loss.backward()
            global_step += 1

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            avg_lm_loss = np.mean(lm_loss) if lm_loss else -1
            avg_mc_loss = np.mean(mc_loss) if mc_loss else -1
            avg_rp_loss = np.mean(rp_loss) if rp_loss else -1
            progress_bar.set_postfix(lm_loss=avg_lm_loss, mc_loss=avg_mc_loss, rp_loss=avg_rp_loss)
            if (step + 1) % logging_steps == 0 or (step + 1) == len(progress_bar):
                logger.info(
                    f"Epoch - {epoch + 1}: {(step + 1) * 100 // len(nli_dataloader):>3}%| "
                    f"lm_loss={avg_lm_loss}, mc_loss={avg_mc_loss}, rp_loss={avg_rp_loss}"
                )
                tqdm.write(
                    f"Epoch - {epoch + 1}: {(step + 1) * 100 // len(nli_dataloader):>3}%| "
                    f"lm_loss={avg_lm_loss}, mc_loss={avg_mc_loss}, rp_loss={avg_rp_loss}"
                )

        lm_loss = []
        mc_loss = []
        rp_loss = []
        logger.info(f"Epoch - {epoch + 1}: Main Task ({data_args.task})")
        logging_steps = int(len(train_dataloader) * model_args.logging_ratio)
        progress_bar = tqdm(train_dataloader, desc=f"Epoch - {epoch + 1}", leave=epoch + 1 == num_train_epochs)
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            result = model(**batch)
            lm_loss.append(result.loss.cpu().detach().item())
            mc_loss.append(result.mc_loss.cpu().detach().item())
            rp_loss.append(result.rp_loss.cpu().detach().item())

            loss = result.loss * model_args.lm_coef + result.mc_loss * model_args.mc_coef + result.rp_loss * model_args.rp_coef

            if training_args.gradient_accumulation_steps > 1:
                loss = loss / training_args.gradient_accumulation_steps

            loss.backward()
            global_step += 1

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            avg_lm_loss = np.mean(lm_loss) if lm_loss else -1
            avg_mc_loss = np.mean(mc_loss) if mc_loss else -1
            avg_rp_loss = np.mean(rp_loss) if rp_loss else -1
            progress_bar.set_postfix(lm_loss=avg_lm_loss, mc_loss=avg_mc_loss, rp_loss=avg_rp_loss)
            if (step + 1) % logging_steps == 0 or (step + 1) == len(progress_bar):
                logger.info(
                    f"Epoch - {epoch + 1}: {(step + 1) * 100 // len(train_dataloader):>3}%| "
                    f"lm_loss={avg_lm_loss}, mc_loss={avg_mc_loss}, rp_loss={avg_rp_loss}"
                )
                tqdm.write(
                    f"Epoch - {epoch + 1}: {(step + 1) * 100 // len(train_dataloader):>3}%| "
                    f"lm_loss={avg_lm_loss}, mc_loss={avg_mc_loss}, rp_loss={avg_rp_loss}"
                )

        logger.info(f"Epoch - {epoch + 1}: Save model!")
        save_dir = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        logger.info(f"Epoch - {epoch + 1}: Begin Evaluate!")
        evaluate(model_args, data_args, training_args, valid_dataset, model, tokenizer, logger, epoch)
        logger.info(f"Epoch - {epoch + 1}: Complete training!")


def evaluate(model_args, data_args, training_args, eval_dataset, model, tokenizer, logger, epoch=None):
    logger.info("Prepare Evaluation Data Loader")
    eval_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset):,}")
    logger.info(f"  Instantaneous batch size per GPU = {training_args.per_device_eval_batch_size}")

    model.eval()
    lm_loss = []
    mc_loss = []
    mc_logits = []
    predictions = []
    responses = []
    with torch.no_grad():
        progress_bar = tqdm(eval_dataloader, desc="Evaluation", leave=epoch is None)
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # PPL, Hits@1
            result = model(**batch)
            lm_loss.append(result.loss.cpu())
            mc_loss.append(result.mc_loss.cpu())
            mc_logits.append(result.mc_logits.cpu())

            # inference is done only during evaluation
            if training_args.predict_with_generate:
                # F1, BLEU, ROUGE
                out_ids = model.generate(
                    input_ids=batch["input_ids"][:, 0, :],
                    attention_mask=batch["attention_mask"][:, 0, :],
                    rp_token_ids=batch["rp_token_ids"][:, 0, :],
                    max_new_tokens=model_args.max_new_tokens,
                    forced_bos_token_id=None,
                    decoder_start_token_id=tokenizer.convert_tokens_to_ids("<resp>"),
                    num_beams=model.generation_config.num_beams,
                    num_return_sequences=model.generation_config.num_beams,
                )   # (bsz * num_beams, seq_len)

                # Select the output with the highest mc_logits from the beam output
                # Construct the input for forwarding through the model again using input and out_ids
                decoder_attention_mask = torch.ones_like(out_ids)
                last_token_index = []
                for out_id, attn_mask in zip(out_ids, decoder_attention_mask):
                    eos_idx = torch.where(out_id == tokenizer.eos_token_id)[0]
                    attn_mask[eos_idx:] = 0
                    last_token_index.append([eos_idx - 1])

                input_ids = batch["input_ids"][:, 0, :].repeat(model.generation_config.num_beams, 1)
                attention_mask = batch["attention_mask"][:, 0, :].repeat(model.generation_config.num_beams, 1)
                decoder_input_ids = out_ids[:, :-1]
                decoder_attention_mask = decoder_attention_mask[:, :-1]
                mc_token_ids = torch.tensor(last_token_index, device=model.device)
                rp_token_ids = batch["rp_token_ids"][:, 0, :].repeat(model.generation_config.num_beams, 1)

                # Forward the newly constructed input through the model again
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    mc_token_ids=mc_token_ids,
                    rp_token_ids=rp_token_ids,
                )

                # Adjust the model's out_ids and max_index to match num_beams and batch_size
                out_ids = out_ids.reshape(-1, model.generation_config.num_beams, out_ids.size(-1))   # (bsz, num_beams, seq_len)
                max_index = result.mc_logits.reshape(-1, model.generation_config.num_beams).argmax(-1)    # (bsz, )

                # Select the output with the highest mc_logits from the beam output
                out_ids = out_ids[torch.arange(out_ids.size(0)), max_index, :]    # (bsz, seq_len)
                prediction = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                response = tokenizer.batch_decode(batch["decoder_input_ids"][:, 0, :], skip_special_tokens=True)

                predictions.extend(prediction)
                responses.extend(response)

            progress_bar.set_postfix(lm_loss=np.mean(lm_loss), mc_loss=np.mean(mc_loss))

    logger.info("Compute metrics")
    average_nll = torch.mean(torch.tensor(lm_loss))
    average_PPL = torch.exp(average_nll)
    mc_logits = torch.cat(mc_logits).reshape(-1, data_args.max_candidates)
    mc_label = torch.zeros(mc_logits.size(0))
    result = {"eval_loss": np.mean(lm_loss)}

    result.update({**compute_metrics_from_logits(data_args, mc_logits, mc_label)})
    result["PPL"] = average_PPL.item()

    # F1, BLEU, ROUGE, C score are calculated only during evaluation
    if training_args.predict_with_generate:
        logger.info("Compute F1, BLEU, ROUGE score")
        parlai_metric = get_parlai_metric(predictions, responses)
        for k, v in parlai_metric.items():
            result[k] = v

        logger.info("Compute C score")
        nli_result, c_score, nli_logits = compute_c_score_from_predictions(model_args, data_args, training_args, predictions, logger)
        result["C"] = c_score

        logger.info(f"Save predictions, responses, and NLI results")
        with open(os.path.join(training_args.output_dir, "test_result.json"), "w") as fp:
            json.dump({"predictions": predictions, "responses": responses, "nli_result": nli_result, "nli_logits": nli_logits}, fp)

    for k, v in result.items():
        result[k] = round(v, 2)

    if epoch is not None:
        result["epoch"] = epoch + 1

    print(result)
    logger.info(f"Eval result: {result}")
    logger.info("Complete Evaulation!")
