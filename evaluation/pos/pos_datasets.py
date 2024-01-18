from pathlib import Path

import conllu
from datasets import Dataset, DatasetDict


class StandardGermanUDDataset:

    def __init__(self):
        self.dataset = self.load_dataset()

    def load_dataset(self, train_path: Path = None, dev_path: Path = None, test_path: Path = None) -> DatasetDict:
        default_dir = Path(__file__).parent.parent.parent / "data" / "UD_German-HDT"
        train_path = train_path or default_dir / "de_hdt-ud-train-a.conllu"
        dev_path = dev_path or default_dir / "de_hdt-ud-dev.conllu"
        test_path = test_path or default_dir / "de_hdt-ud-test.conllu"
        assert train_path.exists()
        assert dev_path.exists()
        assert test_path.exists()
        train_examples = list(self.load_conllu_examples(train_path))
        dev_examples = list(self.load_conllu_examples(dev_path))
        test_examples = list(self.load_conllu_examples(test_path))
        train_dataset = Dataset.from_list(train_examples)
        dev_dataset = Dataset.from_list(dev_examples)
        test_dataset = Dataset.from_list(test_examples)
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": dev_dataset,
            "test": test_dataset,
        })
        dataset = dataset.map(lambda x: {'lang': 'de', **x})
        return dataset

    def print_statistics(self):
        for split, dataset in self.dataset.items():
            print(f"Split: {split}")
            print(f"Number of samples: {len(dataset)}")
            print(f"Number of tokens: {sum(len(example['tokens']) for example in dataset)}")
            print()

    @staticmethod
    def load_conllu_examples(path: Path):
        """
        Source: https://huggingface.co/datasets/universal_dependencies/blob/main/universal_dependencies.py
        """
        id = 0
        with open(path, "r", encoding="utf-8") as data_file:
            tokenlist = list(conllu.parse_incr(data_file))
            for sent in tokenlist:
                if "sent_id" in sent.metadata:
                    idx = sent.metadata["sent_id"]
                else:
                    idx = id

                tokens = [token["form"] for token in sent]

                if "text" in sent.metadata:
                    txt = sent.metadata["text"]
                else:
                    txt = " ".join(tokens)

                yield {
                    # "idx": str(idx),
                    # "text": txt,
                    "tokens": [token["form"] for token in sent if token["upos"] is not None and token["xpos"] is not None],
                    # "lemmas": [token["lemma"] for token in sent],
                    "upos": [token["upos"] for token in sent if token["upos"] is not None and token["xpos"] is not None],
                    "stts": [token["xpos"] for token in sent if token["upos"] is not None and token["xpos"] is not None],
                    # "feats": [str(token["feats"]) for token in sent],
                    # "head": [str(token["head"]) for token in sent],
                    # "deprel": [str(token["deprel"]) for token in sent],
                    # "deps": [str(token["deps"]) for token in sent],
                    # "misc": [str(token["misc"]) for token in sent],
                }
                id += 1


class SwissGermanPOSDataset:

    def __init__(self, type: str = "upos"):
        assert type in {"upos", "stts"}
        self.type = type
        self.dataset = self.load_dataset()

    def load_dataset(self, test_path: Path = None) -> DatasetDict:
        default_dir = Path(__file__).parent.parent.parent / "data" / "GSW_test_set"
        if self.type == "upos":
            test_path = test_path or default_dir / "test_GSW_UPOS.txt"
        elif self.type == "stts":
            test_path = test_path or default_dir / "test_GSW_STTS.txt"
        assert test_path.exists()
        test_examples = list(self.load_examples(test_path))
        test_dataset = Dataset.from_list(test_examples)
        test_dataset = test_dataset.filter(lambda x: len(x["tokens"]) > 0)
        dataset = DatasetDict({
            "test": test_dataset,
        })
        dataset = dataset.map(lambda x: {'lang': 'gsw', **x})
        return dataset

    def load_examples(self, test_path: Path = None):
        with open(test_path) as f:
            tokens = []
            upos = []
            stts = []
            for line in f:
                line = line.strip()
                if not line:
                    if self.type == "upos":
                        yield {
                            "tokens": tokens,
                            "upos": upos,
                        }
                    elif self.type == "stts":
                        yield {
                            "tokens": tokens,
                            "stts": stts,
                        }
                    tokens = []
                    upos = []
                    stts = []
                else:
                    token, pos = line.split()
                    tokens.append(token)
                    if self.type == "upos":
                        upos.append(pos)
                    elif self.type == "stts":
                        stts.append(pos)

    def print_statistics(self):
        for split, dataset in self.dataset.items():
            print(f"Split: {split}")
            print(f"Number of samples: {len(dataset)}")
            print(f"Number of tokens: {sum(len(example['tokens']) for example in dataset)}")
            print()
