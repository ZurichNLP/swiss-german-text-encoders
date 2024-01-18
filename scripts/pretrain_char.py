import subprocess
import sys
from pathlib import Path

from transformers import set_seed

model_name = sys.argv[1]

SEED = 913
NUM_EPOCHS = 20
LEARNING_RATE = "1e-4"

models_dir = Path(__file__).parent.parent / "pretrained_models"
assert models_dir.exists()

data_path = Path(__file__).parent.parent / "data" / "swissdox"
assert data_path.exists()

set_seed(SEED)
output_dir = (models_dir / "de" /
              f"char_{model_name.replace('/', '_')}_v1_{NUM_EPOCHS}e_lr{LEARNING_RATE}_seed{SEED}")
output_dir.mkdir(parents=True, exist_ok=True)

args = [
    "python",
    "-m", "pretraining.run_mlm",
    "--seed", str(SEED),
    "--model_name_or_path", model_name,
    "--train_file", str(data_path / "de_CH.train.txt"),
    "--validation_file", str(data_path / "de_CH.valid.txt"),
    "--do_train",
    "--do_eval",
    "--output_dir", str(output_dir.resolve()),
    "--num_train_epochs", "10",
    "--learning_rate", LEARNING_RATE,
    "--per_device_train_batch_size", "12",
    "--per_device_eval_batch_size", "12",
    "--gradient_accumulation_steps", "16",
    "--logging_steps", "20",
    "--fp16",
    "--preprocessing_num_workers", "10",
    "--dataloader_num_workers", "10",
    "--save_strategy", "epoch",
    "--evaluation_strategy", "epoch",
    "--load_best_model_at_end",
    "--metric_for_best_model", "eval_loss",
    "--auto_find_batch_size",
    "--max_seq_length", "2048",
    "--model_type", "char_xmod",
]

if "xlm-roberta" in model_name.lower():
    raise ValueError("Use the subword script for XLM-R.")
elif "canine" in model_name.lower():
    raise NotImplementedError("No char pre-training necessary for CANINE.")
elif "xmod" in model_name.lower():
    args += ["--adapter_default_language", "de_DE"]
    args += ["--tokenizer_name", "facebook/xmod-base"]
elif "swissbert" in model_name.lower():
    args += ["--adapter_default_language", "de_CH"]
    args += ["--tokenizer_name", "ZurichNLP/swissbert"]

subprocess.run(args)

Path(output_dir / "tokenizer.json").unlink(missing_ok=True)
Path(output_dir / "tokenizer_config.json").unlink(missing_ok=True)
Path(output_dir / "special_tokens_map.json").unlink(missing_ok=True)
