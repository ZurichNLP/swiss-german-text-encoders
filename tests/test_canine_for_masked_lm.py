from unittest import TestCase

from transformers import CanineConfig, AutoTokenizer

from modeling.canine import CanineForMaskedLM


class CanineForMaskedLMTestCase(TestCase):

    def setUp(self):
        self.config = CanineConfig(
           hidden_size=32,
           intermediate_size=37,
           max_position_embeddings=512,
           num_attention_heads=2,
           num_hidden_layers=2,
           vocab_size=32000,
        )
        self.model = CanineForMaskedLM(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained("google/canine-s")

    def test_forward(self):
        M = self.tokenizer.mask_token
        input_ids = self.tokenizer(f"Today I feel {M}{M}{M}{M}{M}.", return_tensors="pt")["input_ids"]
        output = self.model(input_ids, return_dict=True)
        print(output)

    def test_pipeline(self):
        from transformers import pipeline
        fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
        M = self.tokenizer.mask_token
        print(fill_mask(f"Today I read War {M} Peace."))
