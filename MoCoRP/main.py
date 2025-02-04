import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from transformers import AutoTokenizer, Seq2SeqTrainingArguments, HfArgumentParser, set_seed

from models.modeling_mocorp import MoCoRP
from utils.data_utils import build_convai, build_mpchat, build_dnli
from utils.train_model import train, evaluate

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/bart-large")
    nli_model_name_or_path: Optional[str] = field(default=None)
    max_new_tokens: Optional[int] = field(default=50)

    lm_coef: Optional[float] = field(default=1.0)
    mc_coef: Optional[float] = field(default=1.0)
    tp_coef: Optional[float] = field(default=1.0)
    early_stop_epochs: Optional[int] = field(default=None, metadata={"help": "Early stopping epochs."})
    logging_ratio: Optional[float] = field(default=0.05, metadata={"help": "Log every X updates steps."})


@dataclass
class DataArguments:
    dialog_data_dir: Optional[str] = field(default="data/convai2", metadata={"help": "Path to the training data."})
    nli_data_dir: Optional[str] = field(default=None, metadata={"help": "Path to the NLI data. This is for Dialogue Learning"})
    task: Optional[str] = field(default="convai2", metadata={"help": ""})
    original: Optional[bool] = field(default=True, metadata={"help": ""})
    dnli_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the NLI data. This is for Relation Learning"})

    max_history: Optional[int] = field(default=7, metadata={"help": ""})
    num_candidates: Optional[int] = field(default=4, metadata={"help": ""})

    def __post_init__(self):
        if self.task == "convai2":
            self.preprocess_fct = build_convai
            self.train_data_path = os.path.join(self.dialog_data_dir, f"train_self_{'original' if self.original else 'revised'}.txt")
            self.valid_data_path = os.path.join(self.dialog_data_dir, f"valid_self_{'original' if self.original else 'revised'}.txt")
            self.test_data_path = os.path.join(self.dialog_data_dir, f"valid_self_{'original' if self.original else 'revised'}.txt")
            self.max_candidates = 20

            self.train_nli_data_path = os.path.join(self.nli_data_dir, f"train_self_{'original' if self.original else 'revised'}_nli.json")
            self.valid_nli_data_path = os.path.join(self.nli_data_dir, f"valid_self_{'original' if self.original else 'revised'}_nli.json")
            self.test_nli_data_path = None
        else:
            self.preprocess_fct = build_mpchat
            self.train_data_path = os.path.join(self.dialog_data_dir, f"train_mpchat_nrp.json")
            self.valid_data_path = os.path.join(self.dialog_data_dir, f"valid_mpchat_nrp.json")
            self.test_data_path = os.path.join(self.dialog_data_dir, f"test_mpchat_nrp.json")
            self.max_candidates = 100

            self.train_nli_data_path = os.path.join(self.nli_data_dir, f"train_mpchat_nrp_nli.json")
            self.valid_nli_data_path = os.path.join(self.nli_data_dir, f"valid_mpchat_nrp_nli.json")
            self.test_nli_data_path = None

        self.dnli_data_path = os.path.join(self.dnli_data_path, "dialogue_nli_train.jsonl")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    is_main_process = training_args.local_rank in [-1, 0]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # set seed
    set_seed(training_args.seed)

    # Config logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[
            logging.FileHandler(
                os.path.join(
                    training_args.output_dir, "train.log" if not training_args.predict_with_generate else "test.log"
                ),
                mode="w",
            )
        ],
    )

    logger.info("Prepare tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<human>", "<bot>", "<resp>"]})

    logger.info("Prepare model")
    model = MoCoRP.from_pretrained(model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(training_args.device)

    if not training_args.predict_with_generate:
        logger.info("Training mode")
        logger.info("Load NLI dataset")
        nli_dataset = build_dnli(
            data_args.dnli_data_path,
            data_args.task,
            data_args.num_candidates,
            tokenizer,
        )

        logger.info("Load train dataset")
        train_dataset = data_args.preprocess_fct(
            data_args.train_data_path,
            data_args.max_history,
            data_args.num_candidates,
            tokenizer,
            data_args.train_nli_data_path,
        )
        valid_dataset = data_args.preprocess_fct(
            data_args.valid_data_path,
            data_args.max_history,
            data_args.max_candidates,
            tokenizer,
            data_args.valid_nli_data_path,
        )

        logger.info("Begin Training!")
        train(model_args, data_args, training_args, nli_dataset, train_dataset, valid_dataset, model, tokenizer, logger)
    else:
        logger.info("Evaluation mode")
        logger.info("Load dataset for evaluation")
        test_dataset = data_args.preprocess_fct(
            data_args.test_data_path,
            data_args.max_history,
            data_args.max_candidates,
            tokenizer,
            data_args.test_nli_data_path,
        )

        logger.info("Begin Evaluation!")
        evaluate(model_args, data_args, training_args, test_dataset, model, tokenizer, logger)

    logger.info("Process teriminated.")

if __name__ == "__main__":
    main()
