import subprocess
import sys
from pathlib import Path

from transformers import set_seed

model_name = sys.argv[1]

SEED = 913
NUM_EPOCHS = 10
LEARNING_RATE = "1e-4"
FREEZE_CORE_TRANSFORMER = False
FREEZE_EMBEDDINGS = False
LIMIT_TO_GSW = False

models_dir = Path(__file__).parent.parent / "pretrained_models"
assert models_dir.exists()

data_path = Path(__file__).parent.parent / "data" / "continued_pretraining"
assert data_path.exists()

set_seed(SEED)
output_dir = (models_dir / "gsw_de" /
              f"{model_name.replace('/', '_')}_v2{'_full' if not FREEZE_CORE_TRANSFORMER else ''}"
              f"{'_frozen_embeddings' if FREEZE_EMBEDDINGS else ''}"
              f"{'_gsw_only' if LIMIT_TO_GSW else ''}"
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
    "--max_seq_length", "512",
]

if FREEZE_EMBEDDINGS:
    args += ["--freeze_embeddings"]

if LIMIT_TO_GSW:
    args += ["--limit_to_languages", "gsw"]

if "xlm-roberta" in model_name.lower():
    if FREEZE_CORE_TRANSFORMER:
        raise NotImplementedError()
elif "xmod" in model_name.lower():
    args += ["--add_adapters", "de_DE->gsw"]
    if FREEZE_CORE_TRANSFORMER:
        args += ["--freeze_core_transformer"]
elif "swissbert" in model_name.lower():
    args += ["--add_adapters", "de_CH->gsw"]
    if FREEZE_CORE_TRANSFORMER:
        args += ["--freeze_core_transformer"]
elif "canine" in model_name.lower():
    raise ValueError("Use the char script for CANINE.")

subprocess.run(args)
