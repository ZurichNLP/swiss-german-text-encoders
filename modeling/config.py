from transformers import PretrainedConfig, AutoConfig


class CharXmodConfig(PretrainedConfig):
    """
    Combines XmodConfig and CanineConfig.

    `vocab_size` is the size of the subword vocab used for the CANINE-S objective.

    Args:
    char_vocab_size (`int`, *optional*, defaults to 260):
        Vocabulary size of the character/byte embeddings
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the subword embeddings used for the CANINE-S objective.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 16384):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling [`XmodModel`].
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
        For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.
    pre_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization before each block.
    adapter_reduction_factor (`int` or `float`, *optional*, defaults to 2):
        The factor by which the dimensionality of the adapter is reduced relative to `hidden_size`.
    adapter_layer_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply a new layer normalization before the adapter modules (shared across all adapters).
    adapter_reuse_layer_norm (`bool`, *optional*, defaults to `True`):
        Whether to reuse the second layer normalization and apply it before the adapter modules as well.
    ln_before_adapter (`bool`, *optional*, defaults to `True`):
        Whether to apply the layer normalization before the residual connection around the adapter module.
    languages (`Iterable[str]`, *optional*, defaults to `["en_XX"]`):
        An iterable of language codes for which adapter modules should be initialized.
    default_language (`str`, *optional*):
        Language code of a default language. It will be assumed that the input is in this language if no language
        codes are explicitly passed to the forward method.
    downsampling_rate (`int`, *optional*, defaults to 4):
        The rate at which to downsample the original character sequence length before applying the deep Transformer
        encoder.
    upsampling_kernel_size (`int`, *optional*, defaults to 4):
        The kernel size (i.e. the number of characters in each window) of the convolutional projection layer when
        projecting back from `hidden_size`*2 to `hidden_size`.
    local_transformer_stride (`int`, *optional*, defaults to 128):
        The stride of the local attention of the first shallow Transformer encoder. Defaults to 128 for good
        TPU/XLA memory alignment.
    """
    model_type = "char_xmod"

    def __init__(
        self,
        char_vocab_size=261,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=259,
        eos_token_id=1,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        pre_norm=False,
        adapter_reduction_factor=2,
        adapter_layer_norm=False,
        adapter_reuse_layer_norm=True,
        ln_before_adapter=True,
        languages=("en_XX",),
        default_language=None,
        downsampling_rate=4,
        upsampling_kernel_size=4,
        local_transformer_stride=128,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.char_vocab_size = char_vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.pre_norm = pre_norm
        self.adapter_reduction_factor = adapter_reduction_factor
        self.adapter_layer_norm = adapter_layer_norm
        self.adapter_reuse_layer_norm = adapter_reuse_layer_norm
        self.ln_before_adapter = ln_before_adapter
        self.languages = list(languages)
        self.default_language = default_language
        self.downsampling_rate = downsampling_rate
        self.upsampling_kernel_size = upsampling_kernel_size
        self.local_transformer_stride = local_transformer_stride


AutoConfig.register("char_xmod", CharXmodConfig)
