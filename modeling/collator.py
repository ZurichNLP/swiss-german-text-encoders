import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict, List

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from transformers.data.data_collator import _torch_collate_batch


@dataclass
class CanineDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator customized for CANINE-S:
    - mask subword spans based on a subword tokenizer
    - replace every character in a masked subword span with `mask_token`
    - only set a label for one random character per masked subword span (others are -100 = ignored in the loss)
    - restrict random replacements to subwords that have the same length as the original subword

    We assume that the data are tokenized with `tokenizer`, which should be a `CanineTokenizer`, and that
    `subword_tokenizer` is an `XLMRobertaTokenizer`. The implementation is tightly coupled to the specifics of
    these tokenizers.
    """
    tokenizer: PreTrainedTokenizerBase  # Character tokenizer
    subword_tokenizer: PreTrainedTokenizerBase = None  # Subword tokenizer

    def __post_init__(self):
        super().__post_init__()
        assert self.tokenizer is not None and self.tokenizer.mask_token is not None
        from transformers import CanineTokenizer
        assert isinstance(self.tokenizer, CanineTokenizer)
        from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
        assert isinstance(self.subword_tokenizer, (XLMRobertaTokenizer, XLMRobertaTokenizerFast))

        # Precompute subword lengths in terms of chars
        self.subword_length_bins: Dict[int, List[int]] = defaultdict(list)
        self.subword_to_chars: Dict[int, List[int]] = {}
        for subword, subword_id in self.subword_tokenizer.vocab.items():
            raw_subword = subword.replace("▁", "")
            self.subword_length_bins[len(raw_subword)].append(subword_id)
            self.subword_to_chars[subword_id] = self.tokenizer.encode(raw_subword, add_special_tokens=False)

        self.num_skipped_rows = 0  # For logging

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        # Reconstruct the subwords from the character tokens
        input_text = [self.tokenizer.decode(row, skip_special_tokens=True) for row in inputs.tolist()]
        subword_encoding = self.subword_tokenizer.batch_encode_plus(input_text, add_special_tokens=False, return_offsets_mapping=True)
        subword_inputs = _torch_collate_batch(subword_encoding["input_ids"], self.subword_tokenizer)

        subword_labels = subword_inputs.clone()

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(subword_labels.shape, self.mlm_probability)
        subword_special_tokens_mask = [
            self.subword_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in subword_labels.tolist()
        ]
        subword_special_tokens_mask = torch.tensor(subword_special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(subword_special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        subword_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(subword_labels.shape, 0.8)).bool() & masked_indices
        subword_inputs[indices_replaced] = self.subword_tokenizer.convert_tokens_to_ids(self.subword_tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(subword_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # Select random word later, based on byte span

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        pass

        # Project subword labels to character tokens
        # We need to use a for-loop and perform the projection one subword at a time
        original_inputs = inputs.clone()
        labels = -100 * torch.ones_like(inputs)
        for i in range(subword_inputs.shape[0]):
            for j in range(subword_inputs.shape[1]):

                if subword_inputs[i][j] == self.subword_tokenizer.pad_token_id:
                    break

                if subword_labels[i][j] == -100:
                    continue  # Not masked; no projection needed

                subword_token = subword_encoding.encodings[i].tokens[j]
                offset_mapping = subword_encoding.encodings[i].offsets[j]
                char_start, char_end = offset_mapping
                char_span = input_text[i][char_start:char_end]

                if subword_token == "▁":
                    continue

                # Exclude preceding space from the masked character span
                while char_span and char_span[0] == " ":
                    char_span = char_span[1:]
                    char_start += 1
                if not char_span:
                    continue

                if char_span != subword_token.replace("▁", "").strip():
                    # Mapping error; do not mask this input row
                    inputs[i] = original_inputs[i]
                    labels[i] = -100 * torch.ones_like(inputs[i])
                    self.num_skipped_rows += 1
                    self._warn_skipped_rows()
                    break

                # Apply subword-level input perturbations to character inputs
                if indices_replaced[i][j]:  # Replace with mask token
                    inputs[i][char_start:char_end] = self.tokenizer.mask_token_id
                elif indices_random[i][j]:  # Replace with random subword, using precomputed dictionaries
                    if len(char_span) not in self.subword_length_bins:
                        continue
                    random_subword_id = random.choice(self.subword_length_bins[len(char_span)])
                    random_char_ids = self.subword_to_chars[random_subword_id]
                    inputs[i][char_start:char_end] = torch.tensor(random_char_ids)

                # Activate label for one random character in the masked subword
                random_char_idx = torch.randint(char_start, char_end, (1,)).item()
                labels[i, random_char_idx] = subword_labels[i][j]

        if special_tokens_mask is not None:
            labels[special_tokens_mask] = -100

        return inputs, labels

    def tf_mask_tokens(self, *args, **kwargs):
        raise NotImplementedError

    def numpy_mask_tokens(self, *args, **kwargs):
        raise NotImplementedError

    def _warn_skipped_rows(self):
        if self.num_skipped_rows % 100 == 0:
            logging.warning(f"Skipped {self.num_skipped_rows} rows so far due to mapping errors")


@dataclass
class CharXmodDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator customized for CANINE-S and CharXmod with a byte-level tokenizer:
    - mask subword spans based on a subword tokenizer
    - replace every character in a masked subword span with `mask_token`
    - only set a label for one random character per masked subword span (others are -100 = ignored in the loss)
    - restrict random replacements to subwords that have the same length as the original subword

    We assume that the data are tokenized with `tokenizer`, which should be a `CharXmodTokenizer`, and that
    `subword_tokenizer` is an `XLMRobertaTokenizer`. The implementation is tightly coupled to the specifics of
    these tokenizers.
    """
    tokenizer: PreTrainedTokenizerBase  # Character/byte tokenizer
    subword_tokenizer: PreTrainedTokenizerBase = None  # Subword tokenizer

    def __post_init__(self):
        super().__post_init__()
        assert self.tokenizer is not None and self.tokenizer.mask_token is not None
        from modeling.tokenization import CharXmodTokenizer
        assert isinstance(self.tokenizer, CharXmodTokenizer)
        from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
        assert isinstance(self.subword_tokenizer, (XLMRobertaTokenizer, XLMRobertaTokenizerFast))
        for special_token in self.tokenizer.all_special_tokens:
            assert special_token in self.subword_tokenizer.all_special_tokens

        # Precompute subword lengths in terms of bytes
        self.subword_length_bins: Dict[int, List[int]] = defaultdict(list)
        self.subword_to_bytes: Dict[int, List[int]] = {}
        for subword, subword_id in self.subword_tokenizer.vocab.items():
            byte_ids = self.tokenizer.encode(subword.replace("▁", ""), add_special_tokens=False)
            self.subword_length_bins[len(byte_ids)].append(subword_id)
            self.subword_to_bytes[subword_id] = byte_ids

        self.space_id = self.tokenizer._convert_token_to_id(" ")

        self.num_skipped_rows = 0  # For logging

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        # Reconstruct the subwords from the character tokens
        input_text = [self.tokenizer.decode(row) for row in inputs.tolist()]
        subword_encoding = self.subword_tokenizer.batch_encode_plus(input_text, add_special_tokens=False, return_offsets_mapping=True)
        subword_inputs = _torch_collate_batch(subword_encoding["input_ids"], self.subword_tokenizer)

        subword_labels = subword_inputs.clone()

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(subword_labels.shape, self.mlm_probability)
        subword_special_tokens_mask = [
            self.subword_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in subword_labels.tolist()
        ]
        subword_special_tokens_mask = torch.tensor(subword_special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(subword_special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        subword_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(subword_labels.shape, 0.8)).bool() & masked_indices
        subword_inputs[indices_replaced] = self.subword_tokenizer.convert_tokens_to_ids(self.subword_tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(subword_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # Select random word later, based on byte span

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        pass

        # Project subword labels to character tokens
        # We need to use a for-loop and perform the projection one subword at a time
        original_inputs = inputs.clone()
        labels = -100 * torch.ones_like(inputs)
        for i in range(subword_inputs.shape[0]):
            byte_offset = 0
            for j in range(subword_inputs.shape[1]):
                if subword_inputs[i][j] == self.subword_tokenizer.pad_token_id:
                    break

                subword_token = subword_encoding.encodings[i].tokens[j]
                offset_mapping = subword_encoding.encodings[i].offsets[j]
                char_span = input_text[i][offset_mapping[0]:offset_mapping[1]]  # chars, not bytes!

                # Increase byte offset by the bytes between the previous subword and the current subword (if any)
                if j > 0:
                    prev_char_end_idx = subword_encoding.encodings[i].offsets[j - 1][1]
                    char_gap = input_text[i][prev_char_end_idx:offset_mapping[0]]
                    if char_gap:
                        byte_offset += len(self.tokenizer.encode(char_gap, add_special_tokens=False))

                byte_start = byte_offset
                if byte_start >= inputs.shape[1]:
                    # Mapping error; do not mask this input row
                    inputs[i] = original_inputs[i]
                    labels[i] = -100 * torch.ones_like(inputs[i])
                    self.num_skipped_rows += 1
                    self._warn_skipped_rows()
                    break

                # XLM tokenizer sometimes adds a "▁" token before the first word, ignore
                # Use lambda for lazy evaluation
                if all(condition() for condition in [
                    lambda: j == 0 or j == 1,
                    lambda: subword_token == "▁",
                    lambda: original_inputs[i][byte_start].item() != self.space_id,
                ]):
                    continue

                byte_span = self.tokenizer.encode(char_span, add_special_tokens=False)
                byte_end = byte_start + len(byte_span)
                byte_offset = byte_end

                if subword_labels[i][j] == -100:
                    continue  # Not masked; no projection needed

                if byte_end > inputs.shape[1] or self.tokenizer.decode(inputs[i][byte_start:byte_end]).strip() != subword_token.replace("▁", "").strip():
                    # Mapping error; do not mask this input row
                    inputs[i] = original_inputs[i]
                    labels[i] = -100 * torch.ones_like(inputs[i])
                    self.num_skipped_rows += 1
                    self._warn_skipped_rows()
                    break

                # Exclude preceding space from the masked character span
                while byte_span and byte_span[0] == self.space_id:
                    byte_span.pop(0)
                    byte_start += 1
                if not byte_span:
                    continue

                # Apply subword-level input perturbations to character inputs
                if indices_replaced[i][j]:  # Replace with mask token
                    inputs[i][byte_start:byte_end] = self.tokenizer.mask_token_id
                elif indices_random[i][j]:  # Replace with random subword, using precomputed dictionaries
                    if len(byte_span) not in self.subword_length_bins:
                        continue
                    random_subword_id = random.choice(self.subword_length_bins[len(byte_span)])
                    random_byte_ids = self.subword_to_bytes[random_subword_id]
                    inputs[i][byte_start:byte_end] = torch.tensor(random_byte_ids)

                # Activate label for one random character in the masked subword
                random_char_idx = torch.randint(byte_start, byte_end, (1,)).item()
                labels[i, random_char_idx] = subword_labels[i][j]

        if special_tokens_mask is not None:
            labels[special_tokens_mask] = -100

        return inputs, labels

    def tf_mask_tokens(self, *args, **kwargs):
        raise NotImplementedError

    def numpy_mask_tokens(self, *args, **kwargs):
        raise NotImplementedError

    def _warn_skipped_rows(self):
        if self.num_skipped_rows % 100 == 0:
            logging.warning(f"Skipped {self.num_skipped_rows} rows so far due to mapping errors")
