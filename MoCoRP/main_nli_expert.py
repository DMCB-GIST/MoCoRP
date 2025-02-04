import os
import random
import logging
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.utils_nli_expert import build_dnli
from utils.train_nli_expert import train, evaluate

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required (or pre-defined) params
    parser.add_argument("--dnli_data_dir", type=str, default="data/dialogue_nli")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, help="The output directory where the model checkpoints will be written.")

    # Configs
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # Misc: other params (model, input, etc)
    parser.add_argument("--nli_batch_size", default=64, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--early_stop_epochs", default=-1, type=int, help="Early stopping epochs, -1 means no early stopping.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.03, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")

    args = parser.parse_args()

    if args.do_train and args.do_test:
        raise ValueError("Cannot train and test the model simultaneously")
    elif not args.do_train and not args.do_test:
        raise ValueError("You should either train or test the model")
    if args.do_test:
        args.output_dir, _ = os.path.split(args.model_checkpoint)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 and not args.fp16:
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        raise NotImplementedError

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Config logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "train.log" if args.do_train else "test.log"), mode="w"),
        ]
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16
    )

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # Prepare model
    id2label = {0: "neutral", 1: "positive", 2: "negative"}
    label2id = {"neutral": 0, "positive": 1, "negative": 2}
    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        # load datasets for training & validation
        # load train dataset
        train_dataset = build_dnli(args.dnli_data_dir, tokenizer, mode="train")
        logger.info(f"loaded file from {args.dnli_data_dir}")

        # load valid dataset
        valid_dataset = build_dnli(args.dnli_data_dir, tokenizer, mode="dev")
        logger.info(f"loaded file from {args.dnli_data_dir}")

        logger.info(f"num. of train_dataset: {len(train_dataset):,}")
        logger.info(f"num. of valid_dataset: {len(valid_dataset):,}")

        # Begin Training!
        train(args, train_dataset, valid_dataset, model, tokenizer, logger)
    else:
        # load dataset for evaluation
        test_dataset = build_dnli(args.dnli_data_dir, tokenizer, mode="train")
        logger.info(f"loaded file from {args.dnli_data_dir}")

        logger.info(f"num. of test_dataset: {len(test_dataset):,}")

        # Begin Evaluation!
        evaluate(args, test_dataset, model, tokenizer, logger)


if __name__ == "__main__":
    main()
