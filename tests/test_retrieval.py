from unittest import TestCase

from transformers import AutoModel, AutoTokenizer

from evaluation.retrieval.retrieval_datasets import NTREXDataset
from evaluation.retrieval.run_retrieval import SentenceAlignmentBenchmark, SentenceEncoder


class NTREXDatasetTestCase(TestCase):

    def setUp(self):
        self.dataset = NTREXDataset()

    def test_de(self):
        sentences = self.dataset.get_sentences('deu')
        print(sentences[0])
        self.assertEqual(len(sentences), 1997)

    def test_gsw_be(self):
        sentences = self.dataset.get_sentences("gsw-BE")
        print(sentences[0])
        self.assertEqual(len(sentences), 1997)

    def test_gsw_zh(self):
        sentences = self.dataset.get_sentences("gsw-ZH")
        print(sentences[0])
        self.assertEqual(len(sentences), 1997)

    def test_print_statistics(self):
        for language in ['deu', 'gsw-BE', 'gsw-ZH']:
            print(f"Language: {language}")
            self.dataset.print_statistics(language)


class SentenceEncoderTestCase(TestCase):

    def setUp(self):
        model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        self.encoder = SentenceEncoder(
            model=model,
            tokenizer=tokenizer,
            aggregation="average",
        )

    def test_embed_sentence(self):
        text = "The quick brown fox jumps over the lazy dog."
        embeddings = self.encoder.embed_tokens(text)
        self.assertEqual(embeddings.shape, (12, 128))

    def test_embed_sentence__single(self):
        self.encoder.aggregation = "single"
        text = "The quick brown fox jumps over the lazy dog."
        embeddings = self.encoder.embed_tokens(text)
        self.assertEqual(embeddings.shape, (12, 128))


class CanineSentenceEncoderTestCase(TestCase):

    def setUp(self):
        from modeling.config import CharXmodConfig
        from modeling.model import CharXmodModel
        from modeling.tokenization import CharXmodTokenizer
        self.config = CharXmodConfig(
            hidden_size=32,
            intermediate_size=37,
            languages=["de_CH", "gsw_CH"],
            max_position_embeddings=512,
            num_attention_heads=2,
            num_hidden_layers=2,
            vocab_size=32000,
        )
        model = CharXmodModel(self.config)
        tokenizer = CharXmodTokenizer()
        self.encoder = SentenceEncoder(
            model=model,
            tokenizer=tokenizer,
            aggregation="average",
        )
        self.assertTrue(self.encoder._is_canine)

    def test_embed_sentence(self):
        text = "The quick brown fox jumps over the lazy dog."
        embeddings = self.encoder.embed_tokens(text, lang_id=0)
        self.assertEqual(embeddings.shape, (11, 32))

    def test_embed_sentence__single(self):
        self.encoder.aggregation = "single"
        text = "The quick brown fox jumps over the lazy dog."
        embeddings = self.encoder.embed_tokens(text, lang_id=0)
        self.assertEqual(embeddings.shape, (11, 32))


class SentenceAlignmentBenchmarkTestCase(TestCase):

    def setUp(self):
        self.src_sentences = [
            "This is a test sentence.",
            "This is another test sentence.",
        ]
        self.tgt_sentences = [
            "This is a test sentence.",
            "This is another test sentence.",
        ]
        self.benchmark = SentenceAlignmentBenchmark(self.src_sentences, self.tgt_sentences)

    def test_bert_score(self):
        from transformers import AutoModel
        model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        encoder = SentenceEncoder(
            model=model,
            tokenizer=tokenizer,
        )
        result = self.benchmark.evaluate_encoder_bert_score(encoder)
        self.assertEqual(result.accuracy, 1.0)
        self.benchmark.tgt_sentences = self.tgt_sentences[::-1]
        result = self.benchmark.evaluate_encoder_bert_score(encoder)
        self.assertEqual(result.accuracy, 0.0)
