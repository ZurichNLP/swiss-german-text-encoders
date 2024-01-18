from pathlib import Path
from typing import List

from tokenizers.pre_tokenizers import BertPreTokenizer


class NTREXDataset:

    def __init__(self):
        self.default_dir = Path(__file__).parent.parent.parent / 'data' / 'NTREX'

    def get_sentences(self, lang_code: str, path: Path = None) -> List[str]:
        if path is None:
            path = self.default_dir / "NTREX-128" / f"newstest2019-ref.{lang_code}.txt"
        assert path.exists()
        sentences = path.read_text().splitlines()
        assert len(sentences) == 1997
        return sentences

    def print_statistics(self, lang_code: str, path: Path = None):
        sentences = self.get_sentences(lang_code, path)
        tokenizer = BertPreTokenizer()
        print(f"Number of samples: {len(sentences)}")
        print(f"Number of tokens: {sum(len(tokenizer.pre_tokenize_str(s)) for s in sentences)}")
