import subprocess
import sys
from pathlib import Path

import requests
from transformers import set_seed

model_name = sys.argv[1]

SEED = 913
NUM_EPOCHS = 10
LEARNING_RATE = "1e-4"
FREEZE_CORE_TRANSFORMER = False

models_dir = Path(__file__).parent.parent / "pretrained_models"
assert models_dir.exists()

data_path = Path(__file__).parent.parent / "data" / "continued_pretraining"
assert data_path.exists()

set_seed(SEED)
output_dir = (models_dir / "gsw_de" /
              f"char_{model_name.replace('/', '_')}_v2{'_full' if not FREEZE_CORE_TRANSFORMER else ''}"
              f"_{NUM_EPOCHS}e_lr{LEARNING_RATE}_seed{SEED}")
output_dir.mkdir(parents=True, exist_ok=True)

args = [
    "python",
    "-m", "pretraining.run_mlm",
    "--seed", str(SEED),
    "--model_name_or_path", model_name,
    "--saved_dataset_path", str(data_path.resolve()),
    "--do_train",
    "--do_eval",
    "--line_by_line",
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
]

if "xlm-roberta" in model_name.lower():
    raise ValueError("Use the subword script for XLM-R.")
elif "xmod" in model_name.lower():
    args += ["--model_type", "char_xmod"]
    args += ["--add_adapters", "de_DE->gsw"]
    args += ["--tokenizer_name", "facebook/xmod-base"]
    if FREEZE_CORE_TRANSFORMER:
        args += ["--freeze_core_transformer"]
    print("Initializing character modules from CANINE.")
    args += ["--initialize_xmod_modules_with_canine"]
elif "swissbert" in model_name.lower():
    args += ["--model_type", "char_xmod"]
    args += ["--add_adapters", "de_CH->gsw"]
    args += ["--tokenizer_name", "ZurichNLP/swissbert"]
    if FREEZE_CORE_TRANSFORMER:
        args += ["--freeze_core_transformer"]
    print("Initializing character modules from CANINE.")
    args += ["--initialize_xmod_modules_with_canine"]
elif "canine" in model_name.lower():
    args += ["--tokenizer_name", "ZurichNLP/swissbert"]
    print("Initializing output embeddings from SwissBERT.")
    args += ["--initialize_canine_output_embeddings_with_swissbert"]

subprocess.run(args)

Path(output_dir / "tokenizer.json").unlink(missing_ok=True)
Path(output_dir / "tokenizer_config.json").unlink(missing_ok=True)
Path(output_dir / "special_tokens_map.json").unlink(missing_ok=True)

if "canine" in model_name.lower():
    print("Downloading CANINE tokenizer files.")
    r = requests.get("https://huggingface.co/google/canine-s/raw/main/tokenizer_config.json")
    with open(output_dir / "tokenizer_config.json", "w") as f:
        f.write(r.text)
    r = requests.get("https://huggingface.co/google/canine-s/raw/main/special_tokens_map.json")
    with open(output_dir / "special_tokens_map.json", "w") as f:
        f.write(r.text)
