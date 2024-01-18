import copy
import re
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel, CanineModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput, MaskedLMOutput
from transformers.models.canine.modeling_canine import CanineEncoder, CharactersToMolecules, \
    ConvProjection, CaninePooler, CanineModelOutputWithPooling
from transformers.models.xmod.modeling_xmod import XmodEmbeddings, XmodEncoder, XmodLMHead
from transformers.utils import logging

from modeling.config import CharXmodConfig

logger = logging.get_logger(__name__)


class CharXmodPreTrainedModel(PreTrainedModel):
    config_class = CharXmodConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.canine.modeling_canine.CaninePreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Copied from transformers.models.canine.modeling_canine.CaninePreTrainedModel._set_gradient_checkpointing
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CanineEncoder):
            module.gradient_checkpointing = value

    # Copied from transformers.models.xmod.modeling_xmod.XmodPreTrainedModel.set_default_language
    def set_default_language(self, language: str):
        """
        Set the default language code for the model. This is used when the language is not specified in the input.

        Args:
            language (`str`): The language code, such as `"en_XX"` or `"de_DE"`.
        """
        if language not in self.config.languages:
            raise ValueError(
                f"{self} does not have an adapter for {language}. Supported languages: {list(self.config.languages)}"
            )
        self.config.default_language = language

    def freeze_core_transformer(self):
        """
        Freeze all parameters of the core transformer
        Does not freeze the language adapters and the down/upsampling components, nor any prediction heads.
        """
        logger.info("Freezing core transformer")
        excluded_keys = [
            r"^(?!roberta.encoder).*",
            "roberta.encoder.layer.*.output.adapter_layer_norm.(weight|bias)",
            "roberta.encoder.layer.*.output.adapter_modules.*.(weight|bias)",
        ]
        for key, param in self.named_parameters():
            if any([bool(re.fullmatch(regex, key)) for regex in excluded_keys]):
                continue
            param.requires_grad = False

    def freeze_character_components(self):
        logger.info("Freezing character components")
        keys = [
            r"roberta.char_embeddings.*",
            r"roberta.initial_char_encoder.*",
            r"roberta.chars_to_molecules.*",
            r"roberta.final_char_encoder.*",
            r"roberta.projection.*",
        ]
        for key, param in self.named_parameters():
            if any([bool(re.fullmatch(regex, key)) for regex in keys]):
                param.requires_grad = False

    def freeze_character_encoder(self):
        logger.info("Freezing character encoder")
        keys = [
            r"roberta.char_embeddings.*",
            r"roberta.initial_char_encoder.*",
            r"roberta.chars_to_molecules.*",
        ]
        for key, param in self.named_parameters():
            if any([bool(re.fullmatch(regex, key)) for regex in keys]):
                param.requires_grad = False


class CharXmodModel(CharXmodPreTrainedModel):
    """
    X-MOD model with added CANINE downsampling and upsampling, so that the model can operate on characters.

    We use UTF-8 bytes as input units. This is a difference to the original CANINE, which uses
    Unicode code points. We also modify the CANINE downsampling to use learned embeddings instead of hash embeddings.
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        shallow_config = copy.deepcopy(config)
        shallow_config.num_hidden_layers = 1
        shallow_config.vocab_size = shallow_config.char_vocab_size
        shallow_config.max_position_embeddings *= config.downsampling_rate

        self.encoder = XmodEncoder(config)

        # Downsampling components
        self.char_embeddings = XmodEmbeddings(shallow_config)
        self.initial_char_encoder = CanineEncoder(
            shallow_config,
            local=True,
            always_attend_to_first_position=False,
            first_position_attends_to_all=False,
            attend_from_chunk_width=config.local_transformer_stride,
            attend_from_chunk_stride=config.local_transformer_stride,
            attend_to_chunk_width=config.local_transformer_stride,
            attend_to_chunk_stride=config.local_transformer_stride,
        )
        self.chars_to_molecules = CharactersToMolecules(config)

        # Upsampling components
        self.projection = ConvProjection(config)
        self.final_char_encoder = CanineEncoder(shallow_config)

        self.pooler = CaninePooler(config) if add_pooling_layer else None

        # Subword embeddings for CANINE-S loss
        # Adding this as a module so that it is loaded and saved in the model's state_dict
        self.embeddings = XmodEmbeddings(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # Copied from transformers.models.canine.modeling_canine.CanineModel._create_3d_attention_mask_from_input_mask
    def _create_3d_attention_mask_from_input_mask(self, from_tensor, to_mask):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        batch_size, from_seq_length = from_tensor.shape[0], from_tensor.shape[1]

        to_seq_length = to_mask.shape[1]

        to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        broadcast_ones = torch.ones(size=(batch_size, from_seq_length, 1), dtype=torch.float32, device=to_mask.device)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask

    # Copied from transformers.models.canine.modeling_canine.CanineModel._downsample_attention_mask
    def _downsample_attention_mask(self, char_attention_mask: torch.Tensor, downsampling_rate: int):
        """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""

        # first, make char_attention_mask 3D by adding a channel dim
        batch_size, char_seq_len = char_attention_mask.shape
        poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))

        # next, apply MaxPool1d to get pooled_molecule_mask of shape (batch_size, 1, mol_seq_len)
        pooled_molecule_mask = torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)(
            poolable_char_mask.float()
        )

        # finally, squeeze to get tensor of shape (batch_size, mol_seq_len)
        molecule_attention_mask = torch.squeeze(pooled_molecule_mask, dim=-1)

        return molecule_attention_mask

    # Copied from transformers.models.canine.modeling_canine.CanineModel._repeat_molecules
    def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: torch.Tensor) -> torch.Tensor:
        """Repeats molecules to make them the same length as the char sequence."""

        rate = self.config.downsampling_rate

        molecules_without_extra_cls = molecules[:, 1:, :]
        # `repeated`: [batch_size, almost_char_seq_len, molecule_hidden_size]
        repeated = torch.repeat_interleave(molecules_without_extra_cls, repeats=rate, dim=-2)

        # So far, we've repeated the elements sufficient for any `char_seq_length`
        # that's a multiple of `downsampling_rate`. Now we account for the last
        # n elements (n < `downsampling_rate`), i.e. the remainder of floor
        # division. We do this by repeating the last molecule a few extra times.
        last_molecule = molecules[:, -1:, :]
        remainder_length = torch.fmod(torch.tensor(char_seq_length), torch.tensor(rate)).item()
        remainder_repeated = torch.repeat_interleave(
            last_molecule,
            # +1 molecule to compensate for truncation.
            repeats=remainder_length + rate,
            dim=-2,
        )

        # `repeated`: [batch_size, char_seq_len, molecule_hidden_size]
        return torch.cat([repeated, remainder_repeated], dim=-2)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CanineModelOutputWithPooling]:
        """
        Adapted from transformers.models.canine.modeling_canine.CanineModel.forward.
        We use lang_ids to route the input through the correct adapter, like in
            transformers.models.xmod.modeling_xmod.XmodModel.forward.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if lang_ids is None:
            if self.config.default_language is None:
                raise ValueError("Input language unknown. Please call `CharXmodPreTrainedModel.set_default_language()`")
            adapter_languages = list(self.encoder.layer[0].output.adapter_modules.keys())
            default_lang_id = adapter_languages.index(self.config.default_language)
            lang_ids = default_lang_id * torch.ones(batch_size, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        molecule_attention_mask = self._downsample_attention_mask(
            attention_mask, downsampling_rate=self.config.downsampling_rate
        )
        extended_molecule_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            molecule_attention_mask, (batch_size, molecule_attention_mask.shape[-1])
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # `input_char_embeddings`: shape (batch_size, char_seq, char_dim)
        input_char_embeddings = self.char_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Contextualize character embeddings using shallow Transformer.
        # We use a 3D attention mask for the local attention.
        # `input_char_encoding`: shape (batch_size, char_seq_len, char_dim)
        char_attention_mask = self._create_3d_attention_mask_from_input_mask(input_ids, attention_mask)
        init_chars_encoder_outputs = self.initial_char_encoder(
            input_char_embeddings,
            attention_mask=char_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        input_char_encoding = init_chars_encoder_outputs.last_hidden_state

        # Downsample chars to molecules.
        # The following lines have dimensions: [batch, molecule_seq, molecule_dim].
        # In this transformation, we change the dimensionality from `char_dim` to
        # `molecule_dim`, but do *NOT* add a resnet connection. Instead, we rely on
        # the resnet connections (a) from the final char transformer stack back into
        # the original char transformer stack and (b) the resnet connections from
        # the final char transformer stack back into the deep BERT stack of
        # molecules.
        #
        # Empirically, it is critical to use a powerful enough transformation here:
        # mean pooling causes training to diverge with huge gradient norms in this
        # region of the model; using a convolution here resolves this issue. From
        # this, it seems that molecules and characters require a very different
        # feature space; intuitively, this makes sense.
        init_molecule_encoding = self.chars_to_molecules(input_char_encoding)

        # Deep BERT encoder
        # `molecule_sequence_output`: shape (batch_size, mol_seq_len, mol_dim)
        encoder_outputs = self.encoder(
            init_molecule_encoding,
            lang_ids=lang_ids,
            attention_mask=extended_molecule_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        molecule_sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(molecule_sequence_output) if self.pooler is not None else None

        # Upsample molecules back to characters.
        # `repeated_molecules`: shape (batch_size, char_seq_len, mol_hidden_size)
        repeated_molecules = self._repeat_molecules(molecule_sequence_output, char_seq_length=input_shape[-1])

        # Concatenate representations (contextualized char embeddings and repeated molecules):
        # `concat`: shape [batch_size, char_seq_len, molecule_hidden_size+char_hidden_final]
        concat = torch.cat([input_char_encoding, repeated_molecules], dim=-1)

        # Project representation dimension back to hidden_size
        # `sequence_output`: shape (batch_size, char_seq_len, hidden_size])
        sequence_output = self.projection(concat)

        # Apply final shallow Transformer
        # `sequence_output`: shape (batch_size, char_seq_len, hidden_size])
        final_chars_encoder_outputs = self.final_char_encoder(
            sequence_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = final_chars_encoder_outputs.last_hidden_state

        if output_hidden_states:
            deep_encoder_hidden_states = encoder_outputs.hidden_states if return_dict else encoder_outputs[1]
            all_hidden_states = (
                all_hidden_states
                + init_chars_encoder_outputs.hidden_states
                + deep_encoder_hidden_states
                + final_chars_encoder_outputs.hidden_states
            )

        if output_attentions:
            deep_encoder_self_attentions = encoder_outputs.attentions if return_dict else encoder_outputs[-1]
            all_self_attentions = (
                all_self_attentions
                + init_chars_encoder_outputs.attentions
                + deep_encoder_self_attentions
                + final_chars_encoder_outputs.attentions
            )

        if not return_dict:
            output = (sequence_output, pooled_output)
            output += tuple(v for v in [all_hidden_states, all_self_attentions] if v is not None)
            return output

        return CanineModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def initialize_from_canine(self, canine: CanineModel):
        """
        Initialize the character modules from a pretrained CANINE model.
        """
        assert self.config.hidden_size == canine.config.hidden_size
        self.initial_char_encoder.load_state_dict(canine.initial_char_encoder.state_dict())
        self.chars_to_molecules.load_state_dict(canine.chars_to_molecules.state_dict())
        self.projection.load_state_dict(canine.projection.state_dict())
        self.final_char_encoder.load_state_dict(canine.final_char_encoder.state_dict())


class CharXmodForMaskedLM(CharXmodPreTrainedModel):
    """
    Implements a subword loss, called CANINE-S in the original paper.
    """
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = CharXmodModel(config, add_pooling_layer=False)
        self.lm_head = XmodLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.xmod.modeling_xmod.XmodForMaskedLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # Copied from transformers.models.xmod.modeling_xmod.XmodForMaskedLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def freeze_output_embeddings(self):
        logger.info("Freezing MLM output embeddings")
        self.lm_head.decoder.weight.requires_grad = False

    # Copied from transformers.models.xmod.modeling_xmod.XmodForMaskedLM.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CharXmodForSequenceClassification(CharXmodPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = CharXmodModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # Adapted from transformers.models.canine.modeling_canine.CanineForSequenceClassification.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CharXmodForTokenClassification(CharXmodPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = CharXmodModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # Adapted from transformers.models.canine.modeling_canine.CanineForTokenClassification.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
