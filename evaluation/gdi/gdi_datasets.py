from pathlib import Path

from datasets import load_dataset
from tokenizers.pre_tokenizers import BertPreTokenizer


class GDIVardial2019Dataset:

    def __init__(self, train_path: Path = None, validation_path: Path = None, test_path: Path = None):
        self.default_dir = Path(__file__).parent.parent.parent / 'data' / 'gdi-vardial-2019'
        self.train_path = train_path if train_path else self.default_dir / 'train.txt'
        self.validation_path = validation_path if validation_path else self.default_dir / 'dev.txt'
        self.test_path = test_path if test_path else self.default_dir / 'gold.txt'
        assert self.train_path.exists()
        assert self.validation_path.exists()
        assert self.test_path.exists()
        self.dataset = load_dataset('csv', data_files={
            'train': str(self.train_path),
            'validation': str(self.validation_path),
            'test': str(self.test_path),
        }, delimiter='\t', column_names=['text', 'label'])
        self.dataset = self.dataset.map(lambda x: {'lang': 'gsw', **x})

    def print_statistics(self):
        tokenizer = BertPreTokenizer()
        for split in ['train', 'validation', 'test']:
            print(f'Statistics for {split}:')
            print(f"Number of samples: {len(self.dataset[split])}")
            print(f"Number of tokens: {sum(len(tokenizer.pre_tokenize_str(s['text'])) for s in self.dataset[split])}")
            print(f"Classes: {set(s['label'] for s in self.dataset[split])}")  # {'BE', 'BS', 'ZH', 'LU'}
            print(f"Distribution of classes:")
            for label in ['BE', 'BS', 'ZH', 'LU']:
                print(f"  {label}: {sum(s['label'] == label for s in self.dataset[split])}")
