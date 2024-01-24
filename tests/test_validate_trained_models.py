from unittest import TestCase

import torch
from transformers import AutoModel, AutoTokenizer

from modeling.model import CharXmodModel


class ValidateSwissGermanAdapterTestCase(TestCase):

    def setUp(self):
        self.original_model = AutoModel.from_pretrained("ZurichNLP/swissbert", revision="v1")
        self.original_tokenizer = AutoTokenizer.from_pretrained("ZurichNLP/swissbert", revision="v1")
        self.updated_model = AutoModel.from_pretrained("ZurichNLP/swissbert", revision="v2")
        self.updated_tokenizer = AutoTokenizer.from_pretrained("ZurichNLP/swissbert", revision="v2")

    def test_tokenizer_is_unchanged(self):
        self.assertEqual(self.original_tokenizer.vocab_size, self.updated_tokenizer.vocab_size)
        self.assertEqual(self.original_tokenizer.all_special_tokens, self.updated_tokenizer.all_special_tokens)
        self.assertEqual(self.original_tokenizer.all_special_ids, self.updated_tokenizer.all_special_ids)
        self.assertEqual(self.original_tokenizer.encode("Hello world"), self.updated_tokenizer.encode("Hello world"))

    def test_languages(self):
        self.assertEqual(self.original_model.config.languages, ['de_CH', 'fr_CH', 'it_CH', 'rm_CH'])
        self.assertEqual(list(self.original_model.encoder.layer[0].output.adapter_modules.keys()), self.original_model.config.languages)
        self.assertEqual(self.updated_model.config.languages, ['de_CH', 'fr_CH', 'it_CH', 'rm_CH', 'gsw'])
        self.assertEqual(list(self.updated_model.encoder.layer[0].output.adapter_modules.keys()), self.updated_model.config.languages)

    def test_gsw_adapter_is_updated(self):
        de_adapter = self.updated_model.encoder.layer[0].output.adapter_modules['de_CH']
        gsw_adapter = self.updated_model.encoder.layer[0].output.adapter_modules['gsw']
        self.assertFalse(torch.equal(de_adapter.dense1.weight, gsw_adapter.dense1.weight))
        self.assertFalse(torch.equal(de_adapter.dense2.weight, gsw_adapter.dense2.weight))

    def test_german_adapter_is_unchanged(self):
        de_adapter = self.updated_model.encoder.layer[0].output.adapter_modules['de_CH']
        self.assertTrue(torch.equal(de_adapter.dense1.weight, self.original_model.encoder.layer[0].output.adapter_modules['de_CH'].dense1.weight))
        self.assertTrue(torch.equal(de_adapter.dense2.weight, self.original_model.encoder.layer[0].output.adapter_modules['de_CH'].dense2.weight))

    def test_subword_embeddings_are_unchanged(self):
        self.assertTrue(torch.equal(self.updated_model.embeddings.word_embeddings.weight, self.original_model.embeddings.word_embeddings.weight))

    def test_core_transformer_is_unchanged(self):
        self.assertTrue(torch.equal(self.updated_model.encoder.layer[0].output.dense.weight, self.original_model.encoder.layer[0].output.dense.weight))


class ValidateCharLevelSwissBERTTestCase(TestCase):

    def setUp(self):
        self.updated_model = CharXmodModel.from_pretrained("ZurichNLP/swiss-german-swissbert-char")
        self.original_model = AutoModel.from_pretrained("ZurichNLP/swissbert")

    def test_languages(self):
        self.assertEqual(self.updated_model.config.languages, ['de_CH', 'fr_CH', 'it_CH', 'rm_CH', 'gsw'])
        self.assertEqual(list(self.updated_model.encoder.layer[0].output.adapter_modules.keys()), self.updated_model.config.languages)

    def test_gsw_adapter_is_updated(self):
        de_adapter = self.updated_model.encoder.layer[0].output.adapter_modules['de_CH']
        gsw_adapter = self.updated_model.encoder.layer[0].output.adapter_modules['gsw']
        self.assertFalse(torch.equal(de_adapter.dense1.weight, gsw_adapter.dense1.weight))
        self.assertFalse(torch.equal(de_adapter.dense2.weight, gsw_adapter.dense2.weight))

    def test_core_transformer_is_unchanged(self):
        self.assertTrue(torch.equal(self.updated_model.encoder.layer[0].output.dense.weight, self.original_model.encoder.layer[0].output.dense.weight))

    def test_output_embeddings_are_unchanged(self):
        self.assertTrue(torch.equal(self.updated_model.embeddings.word_embeddings.weight, self.original_model.embeddings.word_embeddings.weight))


class ValidateSwissGermanXLMRTestCase(TestCase):

    def setUp(self):
        self.updated_model = AutoModel.from_pretrained("ZurichNLP/swiss-german-xlm-roberta-base")
        self.updated_tokenizer = AutoTokenizer.from_pretrained("ZurichNLP/swiss-german-xlm-roberta-base")
        self.original_model = AutoModel.from_pretrained("xlm-roberta-base")
        self.original_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def test_tokenizer_is_unchanged(self):
        self.assertEqual(self.original_tokenizer.vocab_size, self.updated_tokenizer.vocab_size)
        self.assertEqual(self.original_tokenizer.all_special_tokens, self.updated_tokenizer.all_special_tokens)
        self.assertEqual(self.original_tokenizer.all_special_ids, self.updated_tokenizer.all_special_ids)
        self.assertEqual(self.original_tokenizer.encode("Hello world"), self.updated_tokenizer.encode("Hello world"))

    def test_core_transformer_is_updated(self):
        self.assertFalse(torch.equal(self.updated_model.encoder.layer[0].output.dense.weight, self.original_model.encoder.layer[0].output.dense.weight))


class ValidateSwissGermanCanineTestCase(TestCase):

    def setUp(self):
        self.updated_model = AutoModel.from_pretrained("ZurichNLP/swiss-german-canine")
        self.original_model = AutoModel.from_pretrained("google/canine-s")

    def test_core_transformer_is_updated(self):
        self.assertFalse(torch.equal(self.updated_model.encoder.layer[0].output.dense.weight, self.original_model.encoder.layer[0].output.dense.weight))
