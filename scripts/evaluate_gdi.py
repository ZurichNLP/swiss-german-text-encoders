import json
import sys
from pathlib import Path
import subprocess
from typing import List, Dict

import numpy as np

from datasets import disable_caching
from transformers import set_seed

sys.path.append(str(Path(__file__).parent.parent.resolve()))

disable_caching()

model_name = sys.argv[1]

FALLBACK_DE = False  # Set to True to use Standard German adapter for Swiss German

SEEDS = [
    553589,
    456178,
    817304,
    6277,
    792418,
]

NUM_EPOCHS = 10
LEARNING_RATE = "2e-5"

models_dir = Path(__file__).parent.parent / "finetuned_models"
assert models_dir.exists()

eval_results: List[Dict] = []
test_results: List[Dict] = []

for seed in SEEDS:
    set_seed(seed)

    output_dir = (models_dir / f"gdi" /
                  f"{model_name.replace('/', '_')}_{NUM_EPOCHS}e_lr{LEARNING_RATE}_seed{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_arguments = [
        "python",
        "-m", "evaluation.gdi.run_gdi",
        "--seed", str(seed),
        "--model_name_or_path", model_name,
        "--dataset_name", "gdi-vardial-2019",
        "--text_column_names", "text",
        "--label_column_name", "label",
        "--output_dir", str(output_dir.resolve()),
        "--do_train",
        "--num_train_epochs", str(NUM_EPOCHS),
        "--do_eval",
        "--metric_name", "f1",
        "--learning_rate", LEARNING_RATE,
        "--per_device_train_batch_size", "16",
        "--max_seq_length", str(4 * 128 if "canine" in model_name else 128),
        "--save_strategy", "epoch",
        "--evaluation_strategy", "epoch",
        "--save_total_limit", "1",
        "--load_best_model_at_end",
        "--metric_for_best_model", "eval_f1",
        "--overwrite_output_dir",
    ]
    if FALLBACK_DE:
        train_arguments += ["--gsw_fallback_de"]
    subprocess.run(train_arguments)
    eval_results_path = output_dir / "eval_results.json"
    with open(eval_results_path, "r") as f:
        results = json.load(f)
    eval_results.append(results)

    # Test
    test_arguments = [
        "python",
        "-m", "evaluation.gdi.run_gdi",
        "--model_name_or_path", str(output_dir.resolve()),
        "--dataset_name", "gdi-vardial-2019",
        "--text_column_names", "text",
        "--label_column_name", "label",
        "--output_dir", str(output_dir.resolve()),
        "--overwrite_output_dir",
        "--do_eval",
        "--validation_split_name", "test",
        "--metric_name", "f1",
    ]
    if FALLBACK_DE:
        test_arguments += ["--gsw_fallback_de"]
    subprocess.run(test_arguments)
    test_results_path = output_dir / "eval_results.json"
    with open(test_results_path, "r") as f:
        results = json.load(f)
    test_results.append(results)


def format_avg_std(values):
    avg = 100 * np.mean(values)
    std = 100 * np.std(values)
    return f"{avg:.1f}\pm{std:.1f}"

print(f"Results for {model_name}")
print("Validation F1: " + format_avg_std([result["eval_f1"] for result in eval_results]))
print("Test F1: " + format_avg_std([result["eval_f1"] for result in test_results]))
