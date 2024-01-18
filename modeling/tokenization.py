import warnings
from typing import List, Optional

from transformers import ByT5Tokenizer, AddedToken, BatchEncoding, CanineTokenizer


class CharXmodTokenizer(ByT5Tokenizer):
    """
    Modifies `ByT5Tokenizer` to behave similar to `XLMRobertaTokenizer`:
    - Includes a `mask_token` <mask>
    - Adds a `cls_token` <s> to the start of the input
    - Adds a `sep_token` </s> between a sentence pair
    """

    def __init__(
            self,
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            extra_ids=0,
            additional_special_tokens=None,
            pre_space=False,  # When grouping chars into words, add spaces to next word instead of previous word
            **kwargs,
    ) -> None:
        super().__init__(
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens or [],
            **kwargs,
        )
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        mask_token = AddedToken(mask_token, lstrip=False, rstrip=False) if isinstance(mask_token, str) else mask_token
        self.add_special_tokens({
            "sep_token": sep_token,
            "cls_token": cls_token,
            "mask_token": mask_token,
        })
        self.clean_up_tokenization_spaces = False
        self.pre_space = pre_space

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return (1 + len(token_ids_0) + 1) * [0]
        return (1 + len(token_ids_0) + 1 + 1 + len(token_ids_1) + 1) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        token_ids_0 = self._add_cls_if_not_present(token_ids_0)
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + [self.sep_token_id] + token_ids_1

    def _add_cls_if_not_present(self, token_ids: List[int]) -> List[int]:
        """
        Adapted from ByT5Tokenizer._add_eos_if_not_present
        """
        if len(token_ids) > 0 and token_ids[0] == self.cls_token_id:
            warnings.warn(
                f"This sequence already has {self.cls_token}. In future versions this behavior may lead to duplicated"
                " cls tokens being added."
            )
            return token_ids
        else:
            return [self.cls_token_id] + token_ids

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Changed default of `spaces_between_special_tokens` to False
        """
        return super()._decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    def get_word_ids(self, encoding: BatchEncoding, batch_index: int = 0) -> List[Optional[int]]:
        """
        Replacement for BatchEncoding.get_word_ids, which is only available for fast tokenizers
        Needed for token classification tasks

        Preceding white space is added to the next word, as in XLM tokenizer
        """
        space_id = self.convert_tokens_to_ids(" ")
        word_ids = []
        word_idx = 0
        if self.pre_space:
            for token_id in encoding.input_ids[batch_index]:
                if token_id in self.all_special_ids:
                    word_ids.append(None)
                elif token_id == space_id:
                    word_idx += 1
                    word_ids.append(word_idx)
                else:
                    word_ids.append(word_idx)
        else:
            for token_id in encoding.input_ids[batch_index]:
                if token_id in self.all_special_ids:
                    word_ids.append(None)
                elif token_id == space_id:
                    word_ids.append(word_idx)
                    word_idx += 1
                else:
                    word_ids.append(word_idx)
        return word_ids

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + [1] + ([0] * len(token_ids_1)) + [1]


class PreSpaceCanineTokenizer(CanineTokenizer):

    def get_word_ids(self, encoding: BatchEncoding, batch_index: int = 0) -> List[Optional[int]]:
        """
        Replacement for BatchEncoding.get_word_ids, which is only available for fast tokenizers
        Needed for token classification tasks

        Preceding white space is added to the next word, as in XLM tokenizer
        """
        space_id = self.convert_tokens_to_ids(" ")
        word_ids = []
        word_idx = 0
        for token_id in encoding.input_ids[batch_index]:
            if token_id in self.all_special_ids:
                word_ids.append(None)
            elif token_id == space_id:
                word_idx += 1
                word_ids.append(word_idx)
            else:
                word_ids.append(word_idx)
        return word_ids


class PostSpaceCanineTokenizer(CanineTokenizer):

    def get_word_ids(self, encoding: BatchEncoding, batch_index: int = 0) -> List[Optional[int]]:
        """
        Replacement for BatchEncoding.get_word_ids, which is only available for fast tokenizers
        Needed for token classification tasks

        White space is added to the preceding word, which might be more natural for token classification
        """
        space_id = self.convert_tokens_to_ids(" ")
        word_ids = []
        word_idx = 0
        for token_id in encoding.input_ids[batch_index]:
            if token_id in self.all_special_ids:
                word_ids.append(None)
            elif token_id == space_id:
                word_ids.append(word_idx)
                word_idx += 1
            else:
                word_ids.append(word_idx)
        return word_ids
