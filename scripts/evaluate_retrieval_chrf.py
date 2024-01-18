import evaluate
from tqdm import tqdm

from evaluation.retrieval.retrieval_datasets import NTREXDataset

chrf = evaluate.load("chrf")

dataset = NTREXDataset()
lang_pairs = [
    ("deu", "gsw-BE"),
    ("deu", "gsw-ZH"),
]

for src_lang, tgt_lang in lang_pairs:
    print(f"Language pair: {src_lang} -> {tgt_lang}")
    src_sentences = dataset.get_sentences(src_lang)
    tgt_sentences = dataset.get_sentences(tgt_lang)

    num_correct = 0
    num_total = 0
    for i, src_sentence in enumerate(tqdm(src_sentences)):
        max_score = 0
        best_j = None
        for j, tgt_sentence in enumerate(tgt_sentences):
            result = chrf.compute(predictions=[src_sentence], references=[[tgt_sentence]])
            score = result["score"]
            if score > max_score:
                max_score = score
                best_j = j
        is_correct = best_j == i
        num_correct += is_correct
        num_total += 1

        if not is_correct:
            print(f"Source: {src_sentence} (i={i})")
            print(f"Target: {tgt_sentences[i]} (i={i})")
            print(f"Prediction: {tgt_sentences[best_j]} (j={best_j})")
            print()
    print(f"Accuracy: {num_correct / num_total}")
