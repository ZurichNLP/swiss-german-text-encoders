from pathlib import Path
import re

from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from tokenizers.pre_tokenizers import BertPreTokenizer


class SwissCrawlDataset:

    def __init__(self, csv_path: Path = None):
        self.csv_path = csv_path or Path(__file__).parent.parent / 'data' / 'swisscrawl' / 'SwissCrawl.csv'
        assert self.csv_path.exists()
        self.dataset = load_dataset('csv', data_files=str(self.csv_path))
        self.dataset = self.dataset.remove_columns(['url', 'crawl_proba', 'date'])
        self.dataset = self.dataset.map(lambda x: {'lang': 'gsw', **x})

    def print_statistics(self):
        tokenizer = BertPreTokenizer()
        print(f"Number of sentences: {len(self.dataset['train'])}")
        print(f"Number of tokens: {sum(len(tokenizer.pre_tokenize_str(sample['text'])) for sample in self.dataset['train'])}")


class SwissGermanTweetsDataset:

    def __init__(self, csv_path: Path = None):
        self.csv_path = csv_path or Path(__file__).parent.parent / 'data' / 'gsw_tweets' / 'gsw_tweets.csv'
        assert self.csv_path.exists()
        self.dataset = load_dataset('csv', data_files=str(self.csv_path), column_names=[
            'tweet_id', 'user_id', 'username', 'text',
        ])
        self.dataset = self.dataset.remove_columns(['tweet_id', 'user_id', 'username'])
        # Replace usernames with "@USER"
        self.dataset = self.dataset.map(lambda x: {'text': re.sub(r'@(\w+)', '@USER', x['text'])})
        # Deduplicate
        pandas_df = self.dataset['train'].to_pandas()
        pandas_df = pandas_df.drop_duplicates(subset=['text'])
        self.dataset = Dataset.from_pandas(pandas_df)
        self.dataset = self.dataset.map(lambda x: {'lang': 'gsw', **x})
        self.dataset = DatasetDict({
            'train': self.dataset,
        })

    def print_statistics(self):
        tokenizer = BertPreTokenizer()
        print(f"Number of tweets: {len(self.dataset['train'])}")
        print(f"Number of tokens: {sum(len(tokenizer.pre_tokenize_str(sample['text'])) for sample in self.dataset['train'])}")


class SwissdoxDataset:

    def __init__(self, train_path: Path = None, valid_path: Path = None):
        default_dir = Path(__file__).parent.parent / 'data' / 'swissdox'
        self.train_path = train_path or default_dir / 'de_CH.train.txt'
        assert self.train_path.exists()
        self.dataset = load_dataset('text', data_files={
            'train': str(self.train_path),
        })
        self.dataset = self.dataset.map(lambda x: {'lang': 'de', **x})

    def print_statistics(self):
        tokenizer = BertPreTokenizer()
        print(f"Number of sentences: {len(self.dataset['train'])}")
        print(f"Number of tokens: {sum(len(tokenizer.pre_tokenize_str(sample['text'])) for sample in self.dataset['train'])}")


class AdaptationDataset:

    def __init__(self, swisscrawl_dataset: SwissCrawlDataset = None, tweets_dataset: SwissGermanTweetsDataset = None, swissdox_dataset: SwissdoxDataset = None):
        self.swisscrawl_dataset = swisscrawl_dataset or SwissCrawlDataset()
        self.tweets_dataset = tweets_dataset or SwissGermanTweetsDataset()
        self.swissdox_dataset = swissdox_dataset or SwissdoxDataset()
        tokenizer = BertPreTokenizer()
        num_gsw_tokens = sum(len(tokenizer.pre_tokenize_str(sample['text'])) for sample in self.swisscrawl_dataset.dataset['train']) + \
                         sum(len(tokenizer.pre_tokenize_str(sample['text'])) for sample in self.tweets_dataset.dataset['train'])
        num_de_tokens = sum(len(tokenizer.pre_tokenize_str(sample['text'])) for sample in self.swissdox_dataset.dataset['train'])
        print(f"Number of tokens in SwissCrawl + Tweets: {num_gsw_tokens}")
        print(f"Original number of tokens in Swissdox: {num_de_tokens}")
        assert num_de_tokens > num_gsw_tokens
        # Subsample swissdox to match num_gsw_tokens
        ratio = (num_de_tokens - num_gsw_tokens) / num_de_tokens
        self.swissdox_dataset.dataset['train'] = self.swissdox_dataset.dataset['train'].train_test_split(test_size=ratio)['train']
        num_de_tokens = sum(len(tokenizer.pre_tokenize_str(sample['text'])) for sample in self.swissdox_dataset.dataset['train'])
        print(f"Subsampled number of tokens in Swissdox: {num_de_tokens}")
        self.dataset = concatenate_datasets([
            self.swisscrawl_dataset.dataset['train'],
            self.tweets_dataset.dataset['train'],
            self.swissdox_dataset.dataset['train'],
        ])

        # Create 5% validation set
        self.dataset = self.dataset.train_test_split(test_size=0.05)
        self.dataset["validation"] = self.dataset["test"]

    def print_statistics(self):
        tokenizer = BertPreTokenizer()
        for split in ['train', 'validation']:
            for lang in ['gsw', 'de']:
                print(f"Number of {lang} sentences in {split}: {len(self.dataset[split].filter(lambda x: x['lang'] == lang))}")
                print(f"Number of {lang} tokens in {split}: {sum(len(tokenizer.pre_tokenize_str(sample['text'])) for sample in self.dataset[split].filter(lambda x: x['lang'] == lang))}")
