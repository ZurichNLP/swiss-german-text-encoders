from pathlib import Path
from unittest import TestCase

from transformers import AutoTokenizer

from modeling.tokenization import CharXmodTokenizer, PreSpaceCanineTokenizer, PostSpaceCanineTokenizer


class CharXmodTokenizerTestCase(TestCase):

    def setUp(self) -> None:
        self.tokenizer = CharXmodTokenizer()

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.mask_token, "<mask>")
        self.assertEqual(self.tokenizer.sep_token, "</s>")
        self.assertEqual(self.tokenizer.cls_token, "<s>")
        self.assertIn(self.tokenizer.mask_token, self.tokenizer.all_special_tokens)
        self.assertIn(self.tokenizer.sep_token, self.tokenizer.all_special_tokens)
        self.assertIn(self.tokenizer.cls_token, self.tokenizer.all_special_tokens)
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.eos_token_id, 1)
        self.assertEqual(self.tokenizer.cls_token_id, 259)
        self.assertEqual(self.tokenizer.tokenize("Hello"), ["H", "e", "l", "l", "o"])
        self.assertEqual(self.tokenizer.tokenize("foo bar"), ["f", "o", "o", " ", "b", "a", "r"])
        self.assertEqual(self.tokenizer.tokenize("foo <mask> bar"), ["f", "o", "o", " ", "<mask>", " ", "b", "a", "r"])
        self.assertEqual(self.tokenizer.tokenize("foo <mask>bar"), ["f", "o", "o", " ", "<mask>", "b", "a", "r"])
        self.assertEqual(self.tokenizer.tokenize("foo<mask> bar"), ["f", "o", "o", "<mask>", " ", "b", "a", "r"])
        self.assertEqual(self.tokenizer.tokenize("foo<mask>bar"), ["f", "o", "o", "<mask>", "b", "a", "r"])

    def test_encode(self):
        encoding = self.tokenizer.encode("This is a test.")
        tokens = [self.tokenizer.decode([token_id]) for token_id in encoding]
        self.assertListEqual(tokens, ["<s>", "T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "t", "e", "s", "t", ".", "</s>"])

    def test_encode_pair(self):
        encoding = self.tokenizer.encode("This is a test.", "This is another test.")
        tokens = [self.tokenizer.decode([token_id]) for token_id in encoding]
        self.assertListEqual(tokens, ["<s>", "T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "t", "e", "s",
                                  "t", ".", "</s>", "</s>", "T", "h", "i", "s", " ", "i", "s", " ", "a", "n", "o",
                                  "t", "h", "e", "r", " ", "t", "e", "s", "t", ".", "</s>"])

    def test_padding(self):
        self.assertEqual(self.tokenizer.pad_token, "<pad>")
        self.assertEqual(self.tokenizer.pad_token, self.tokenizer.decode([self.tokenizer.pad_token_id]))
        encoding = self.tokenizer.batch_encode_plus([
            "This is a sentence.",
            "This is another sentence, which is longer."
        ])
        batch = self.tokenizer.pad(encoding, return_tensors="pt")
        pad_token_id_used = batch["input_ids"][0][-1].item()
        self.assertEqual(pad_token_id_used, self.tokenizer.pad_token_id)
        swissbert_tokenizer = AutoTokenizer.from_pretrained("ZurichNLP/swissbert")
        swissbert_batch = swissbert_tokenizer.pad(encoding, return_tensors="pt")
        swissbert_pad_token_id = swissbert_batch["input_ids"][1][-1].item()
        pad_token_used = self.tokenizer.decode([pad_token_id_used])
        swissbert_pad_token = swissbert_tokenizer.decode([swissbert_pad_token_id])
        self.assertEqual(pad_token_used, swissbert_pad_token)

    def test_decode(self):
        for s in [
            "foo bar",
            "föö bär",
            "<s>foo bar</s>",
            "<s>foo bar",
            "<s> foo bar </s>",
        ]:
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(s, add_special_tokens=False)), s)

    def test_vocab(self):
        self.assertEqual(len(self.tokenizer), 261)

    def test_get_word_ids(self):
        encoding = self.tokenizer.batch_encode_plus(["This is a test."])
        tokens = [self.tokenizer.decode([token_id]) for token_id in encoding["input_ids"][0]]
        self.assertEqual(tokens, ["<s>", "T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "t", "e", "s", "t", ".", "</s>"])
        word_ids = self.tokenizer.get_word_ids(encoding)
        self.assertEqual(word_ids, [None, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, None])

    def test_get_word_ids__pre_space(self):
        self.tokenizer.pre_space = True
        encoding = self.tokenizer.batch_encode_plus(["This is a test."])
        tokens = [self.tokenizer.decode([token_id]) for token_id in encoding["input_ids"][0]]
        self.assertEqual(tokens, ["<s>", "T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "t", "e", "s", "t", ".", "</s>"])
        word_ids = self.tokenizer.get_word_ids(encoding)
        self.assertEqual(word_ids, [None, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, None])

    def test_get_special_tokens_mask(self):
        encoding = self.tokenizer.batch_encode_plus(["This is a test."], return_special_tokens_mask=True)
        self.assertEqual(len(encoding.special_tokens_mask[0]), len(encoding.input_ids[0]))
        self.assertEqual(encoding.special_tokens_mask, [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])


class PreSpaceCanineTokenizerTestCase(TestCase):

    def setUp(self) -> None:
        self.tokenizer = PreSpaceCanineTokenizer.from_pretrained("google/canine-s")

    def test_get_word_ids(self):
        encoding = self.tokenizer.batch_encode_plus(["This is a test."])
        tokens = [self.tokenizer.decode([token_id]) for token_id in encoding["input_ids"][0]]
        self.assertEqual(tokens, [self.tokenizer.cls_token, "T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "t", "e", "s", "t", ".", self.tokenizer.sep_token])
        word_ids = self.tokenizer.get_word_ids(encoding)
        self.assertEqual(word_ids, [None, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, None])


class PostSpaceCanineTokenizerTestCase(TestCase):

    def setUp(self) -> None:
        self.tokenizer = PostSpaceCanineTokenizer.from_pretrained("google/canine-s")

    def test_get_word_ids(self):
        encoding = self.tokenizer.batch_encode_plus(["This is a test."])
        tokens = [self.tokenizer.decode([token_id]) for token_id in encoding["input_ids"][0]]
        self.assertEqual(tokens, [self.tokenizer.cls_token, "T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "t", "e", "s", "t", ".", self.tokenizer.sep_token])
        word_ids = self.tokenizer.get_word_ids(encoding)
        self.assertEqual(word_ids, [None, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, None])


class SwissGermanSubwordTokenizerTestCase(TestCase):

    def setUp(self) -> None:
        tokenizer_path = Path(__file__).parent.parent / "gsw_tokenizer"
        self.assertTrue(tokenizer_path.exists())
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.swissbert_tokenizer = AutoTokenizer.from_pretrained("ZurichNLP/swissbert")

    def test_vocab_size(self):
        self.assertEqual(len(self.tokenizer), len(self.swissbert_tokenizer))

    def test_most_frequent_subwords(self):
        for i in range(100):
            print(i, self.tokenizer._convert_id_to_token(i))
