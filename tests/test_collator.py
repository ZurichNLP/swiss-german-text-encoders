from pathlib import Path
from unittest import TestCase

from transformers import AutoTokenizer, CanineTokenizer

from modeling.collator import CharXmodDataCollatorForLanguageModeling, CanineDataCollatorForLanguageModeling
from modeling.tokenization import CharXmodTokenizer


class CanineDataCollatorForLanguageModelingTestCase(TestCase):

    def setUp(self) -> None:
        self.subword_tokenizer = AutoTokenizer.from_pretrained("ZurichNLP/swissbert")
        self.char_tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
        self.collator = CanineDataCollatorForLanguageModeling(
            tokenizer=self.char_tokenizer,
            subword_tokenizer=self.subword_tokenizer,
            mlm_probability=0.5,
        )

    def test_subword_length_bins(self):
        for subword_id in self.collator.subword_length_bins[16]:
            subword = self.subword_tokenizer.convert_ids_to_tokens(subword_id)
            self.assertEqual(len(self.char_tokenizer.tokenize(subword.replace("▁", ""))), 16)
            print(subword)

    def test_call(self):
        input_text = [
            "This is a test with a few words, including lengthier ones.",
            "A second row."
        ]
        input_ids = self.char_tokenizer.batch_encode_plus(input_text, add_special_tokens=False)["input_ids"]
        for i in range(5):
            batch = self.collator(input_ids)
            print(f"Input chars: {self.char_tokenizer.decode(batch['input_ids'][0])}")
            print(f"Labels: {batch['labels'][0]}")
            print()


class CharXmodDataCollatorForLanguageModelingTestCase(TestCase):

    def setUp(self) -> None:
        self.subword_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.char_tokenizer = CharXmodTokenizer()
        self.collator = CharXmodDataCollatorForLanguageModeling(
            tokenizer=self.char_tokenizer,
            subword_tokenizer=self.subword_tokenizer,
            mlm_probability=0.5,
        )

    def test_subword_length_bins(self):
        for subword_id in self.collator.subword_length_bins[16]:
            subword = self.subword_tokenizer.convert_ids_to_tokens(subword_id)
            self.assertEqual(len(self.char_tokenizer.tokenize(subword.replace("▁", ""))), 16)
            print(subword)

    def test_call(self):
        input_text = [
            "<s>This is a test with a few words, including lengthier ones.</s>",
            "<s>A second row.</s>"
        ]
        input_ids = self.char_tokenizer.batch_encode_plus(input_text, add_special_tokens=False)["input_ids"]
        for i in range(5):
            batch = self.collator(input_ids)
            print(f"Input chars: {self.char_tokenizer.decode(batch['input_ids'][0])}")
            print(f"Labels: {batch['labels'][0]}")
            print()


class SwissBertCharXmodDataCollatorForLanguageModelingTestCase(TestCase):

    def setUp(self) -> None:
        self.subword_tokenizer = AutoTokenizer.from_pretrained("ZurichNLP/swissbert")
        self.char_tokenizer = CharXmodTokenizer()
        self.collator = CharXmodDataCollatorForLanguageModeling(
            tokenizer=self.char_tokenizer,
            subword_tokenizer=self.subword_tokenizer,
        )

    def test_masking(self):
        input_text = ["Dies ist ein kleiner Test."]
        input_ids = self.char_tokenizer.batch_encode_plus(input_text)["input_ids"]
        for i in range(20):
            batch = self.collator(input_ids)
            for j in range(len(batch["input_ids"][0])):
                print(self.char_tokenizer.decode(batch["input_ids"][0][j]), end="\t")
                label = batch["labels"][0][j]
                if label == -100:
                    print()
                else:
                    print(self.subword_tokenizer.decode([label]))
            print()
