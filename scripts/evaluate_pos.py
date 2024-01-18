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
LABEL_TYPE = "stts"

models_dir = Path(__file__).parent.parent / "finetuned_models"
assert models_dir.exists()

eval_results: List[Dict] = []
predict_results: List[Dict] = []

for seed in SEEDS:
    set_seed(seed)

    output_dir = (models_dir / f"pos_{LABEL_TYPE}" /
                  f"{model_name.replace('/', '_')}_{NUM_EPOCHS}e_lr{LEARNING_RATE}_seed{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train on Standard German data
    subprocess.run([
        "python",
        "-m", "evaluation.pos.run_pos",
        "--seed", str(seed),
        "--model_name_or_path", model_name,
        "--dataset_name", "UD_German-HDT",
        "--text_column_name", "tokens",
        "--label_column_name", LABEL_TYPE,
        "--output_dir", str(output_dir.resolve()),
        "--do_train",
        "--num_train_epochs", str(NUM_EPOCHS),
        "--do_eval",
        "--preprocessing_num_workers", "1",
        "--learning_rate", LEARNING_RATE,
        "--per_device_train_batch_size", "16",
        "--save_strategy", "epoch",
        "--evaluation_strategy", "epoch",
        "--save_total_limit", "1",
        "--load_best_model_at_end",
        "--metric_for_best_model", "eval_accuracy",
        "--overwrite_output_dir",
    ])
    eval_results_path = output_dir / "eval_results.json"
    with open(eval_results_path, "r") as f:
        results = json.load(f)
    eval_results.append(results)

    # Test on Swiss German data
    test_arguments = [
        "python",
        "-m", "evaluation.pos.run_pos",
        "--model_name_or_path", str(output_dir.resolve()),
        "--dataset_name", "GSW_test_set",
        "--text_column_name", "tokens",
        "--label_column_name", LABEL_TYPE,
        "--output_dir", str((output_dir / "gsw_test").resolve()),
        "--overwrite_output_dir",
        "--do_predict",
        "--preprocessing_num_workers", "1",
    ]
    if FALLBACK_DE:
        test_arguments += ["--gsw_fallback_de"]
    subprocess.run(test_arguments)
    predict_results_path = output_dir / "gsw_test" / "predict_results.json"
    with open(predict_results_path, "r") as f:
        results = json.load(f)
    predict_results.append(results)


def format_avg_std(values):
    avg = 100 * np.mean(values)
    std = 100 * np.std(values)
    return f"{avg:.1f}\pm{std:.1f}"

print(f"Results for {model_name}")
print("Validation accuracy: " + format_avg_std([result["eval_accuracy"] for result in eval_results]))
print("Test accuracy: " + format_avg_std([result["predict_accuracy"] for result in predict_results]))
