# Adapted from https://github.com/ZurichNLP/swissbert/blob/master/pretraining/tokenization.py

import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import sentencepiece as spm

from pretraining.pretraining_datasets import AdaptationDataset


def create_spm_vocabulary(
        txt_paths: List[Path],
        name: str,
        sampling_alpha: float = 0.3,
        vocab_size: int = 50260,
        user_defined_symbols: List[str] = None,
        tmp_dir: Path = None,
        subsampling_ratio: float = 1.,
):
    for path in txt_paths:
        assert path.exists()
    if tmp_dir is not None:
        tmp_dir = Path(tmp_dir)
        assert tmp_dir.exists()

    print("Counting lines")
    num_lines_orig = []
    # https://stackoverflow.com/a/9631635/3902795
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b
    for path in txt_paths:
        num_lines = 0
        with open(path, "r", encoding="utf-8", errors='ignore') as f:
            num_lines += sum(bl.count("\n") for bl in blocks(f))
        num_lines_orig.append(num_lines / 2)

    num_lines_orig = np.array(num_lines_orig)
    p_orig = num_lines_orig / num_lines_orig.sum()
    p_smooth = p_orig ** sampling_alpha
    p_smooth /= p_smooth.sum()
    num_lines_smooth = p_smooth / max(p_smooth) * max(num_lines_orig)
    expected_repetitions = num_lines_smooth / num_lines_orig
    assert (expected_repetitions >= 1).all()

    print(f"Number of articles per original file: {num_lines_orig}")
    print(f"Original proportions: {p_orig}")
    print(f"Smoothened probabilities: {p_smooth}")
    print(f"Number of articles smoothened: {num_lines_smooth}")
    print(f"Expected repetitions: {expected_repetitions}")

    expected_repetitions *= subsampling_ratio
    print(f"Expected repetitions after subsampling: {expected_repetitions}")

    tmp_in = tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir)
    print(f"Writing lines to {tmp_in.name}")
    for txt_path, rep in zip(txt_paths, expected_repetitions):
        num_lines = 0
        with open(txt_path) as f:
            for line in f:
                if not line.strip():
                    continue
                full, remainder = divmod(rep, 1)
                for _ in range(int(full)):
                    tmp_in.write(line)
                    num_lines += 1
                if np.random.rand() < remainder:
                    tmp_in.write(line)
                    num_lines += 1
        print(f"{num_lines}, ", end="")
    print()

    spm.SentencePieceTrainer.Train(
        f'--user_defined_symbols={",".join(user_defined_symbols) if user_defined_symbols is not None else ""} '
        f'--input={tmp_in.name} '
        '--input_format=text '
        f'--model_prefix={name} '
        f'--vocab_size={vocab_size} '
        '--num_threads=40 '
        '--train_extremely_large_corpus=true '
        '--input_sentence_size=10000000 '
        '--shuffle_input_sentence=true '
    )

    tmp_in.close()
    Path(tmp_in.name).unlink()


if __name__ == "__main__":
    dataset = AdaptationDataset()
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        for sample in dataset.dataset['train']:
            f.write(sample['text'] + "\n")

    create_spm_vocabulary(
        txt_paths=[Path(f.name)],
        name="swissbert_gsw",
        sampling_alpha=0.3,
        vocab_size=50260,
        user_defined_symbols=["</s>"],
    )
    
    model_path = Path("swissbert_gsw.model")
    vocab_path = Path("swissbert_gsw.vocab")
    assert model_path.exists()
    assert vocab_path.exists()
    vocab_dir = Path(__file__).parent.parent / "vocab"
    vocab_dir.mkdir(exist_ok=True)
    shutil.move(model_path, vocab_dir / model_path.name)
    shutil.move(vocab_path, vocab_dir / vocab_path.name)
    Path(f.name).unlink()
