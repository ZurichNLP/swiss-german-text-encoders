from unittest import TestCase

import torch
from transformers import AutoModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.canine.modeling_canine import CanineModelOutputWithPooling

from modeling.config import CharXmodConfig
from modeling.model import CharXmodModel, CharXmodForMaskedLM
from modeling.utils import log_trainable_parameters


class CharXmodModelTestCase(TestCase):

    def setUp(self) -> None:
        self.config = CharXmodConfig(
            hidden_size=32,
            intermediate_size=37,
            languages=["de_CH", "gsw_CH"],
            max_position_embeddings=512,
            num_attention_heads=2,
            num_hidden_layers=2,
            vocab_size=32000,
        )
        self.model = CharXmodModel(self.config)

    def test_print_model(self):
        print(self.model)

    def test_forward(self):
        input_ids = self.model.dummy_inputs["input_ids"]
        self.assertEqual(input_ids.shape, (3, 5))
        lang_ids = torch.Tensor(input_ids.shape[0] * [0]).long()
        output: CanineModelOutputWithPooling = self.model(
            input_ids,
            lang_ids=lang_ids,
            return_dict=True,
        )
        self.assertEqual(output.last_hidden_state.shape, (3, 5, 32))
        self.assertEqual(output.pooler_output.shape, (3, 32))


class CharXmodForMaskedLMTestCase(TestCase):

    def setUp(self) -> None:
        self.config = CharXmodConfig(
            hidden_size=32,
            intermediate_size=37,
            languages=["de_CH", "gsw_CH"],
            max_position_embeddings=512,
            num_attention_heads=2,
            num_hidden_layers=2,
            vocab_size=250002,
        )
        self.model = CharXmodForMaskedLM(self.config)

    def test_forward(self):
        input_ids = self.model.dummy_inputs["input_ids"]
        self.assertEqual(input_ids.shape, (3, 5))
        labels = -100 * torch.ones_like(input_ids)
        labels[0, 1] = input_ids[0, 1]
        labels[1, 2] = input_ids[1, 2]
        labels[2, 3] = input_ids[2, 3]
        lang_ids = torch.Tensor(input_ids.shape[0] * [0]).long()
        model = self.model(input_ids, lang_ids=lang_ids, labels=labels, return_dict=True, )
        output: MaskedLMOutput = model
        self.assertEqual(output.logits.shape, (3, 5, 250002))
        self.assertGreater(output.loss.item(), 0.0)

    def test_print_model(self):
        print(self.model)

    def test_initialize_from_xmod(self):
        model_name_or_path = "hf-internal-testing/tiny-random-XmodForMaskedLM"
        model = CharXmodForMaskedLM.from_pretrained(model_name_or_path)
        print(model)

    def test_freeze_core_transformer(self):
        self.model.freeze_core_transformer()
        print("Frozen parameters:")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(name)
        print()
        print("Unfrozen parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

    def test_freeze_character_components(self):
        self.model.freeze_character_components()
        print("Frozen parameters:")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(name)
        print()
        print("Unfrozen parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

    def test_freeze_output_embeddings(self):
        self.model.freeze_output_embeddings()
        print("Frozen parameters:")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(name)
        print()
        print("Unfrozen parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)


class FullSizeCharXmodForMaskedLMTestCase(TestCase):

    def setUp(self) -> None:
        self.config = CharXmodConfig(
            languages=["de_CH"],
        )
        self.model = CharXmodForMaskedLM(self.config)

    def test_log_trainable_parameters(self):
        log_trainable_parameters(self.model)

    def test_initialize_from_swissbert(self):
        model = CharXmodForMaskedLM.from_pretrained("ZurichNLP/swissbert")

    def test_initialize_from_canine(self):
        model = CharXmodForMaskedLM.from_pretrained("ZurichNLP/swissbert")
        canine = AutoModel.from_pretrained("google/canine-s")
        model.roberta.initialize_from_canine(canine)
