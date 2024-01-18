from pathlib import Path

import datasets
from transformers import AutoTokenizer

tokenizers = {
    "xlmr": AutoTokenizer.from_pretrained("xlm-roberta-base"),
    "swissbert": AutoTokenizer.from_pretrained("ZurichNLP/swissbert"),
    "custom": AutoTokenizer.from_pretrained(Path(__file__).parent.parent / "gsw_tokenizer"),
}

dataset = datasets.load_from_disk(str(Path(__file__).parent.parent / "data" / "continued_pretraining"))
dataset = dataset["train"].select(range(0, 10000))

for name, tokenizer in tokenizers.items():
    print(name)
    print(f"Vocab size: {tokenizer.vocab_size}")
    num_chars = 0
    for example in dataset:
        num_chars += len(example["text"])
    print(f"Num chars in sample: {num_chars}")
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=False, padding=False))
    num_tokens = 0
    for example in tokenized_dataset:
        num_tokens += len(example["input_ids"])
    print(f"Num tokens in sample: {num_tokens}")
    print(f"Compression ratio: {num_chars / num_tokens}")
    print()
