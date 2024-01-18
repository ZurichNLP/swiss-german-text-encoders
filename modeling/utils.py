
def num_parameters(*modules, only_trainable=False):
    return sum \
        (sum(p.numel() for p in module.parameters() if p.requires_grad or not only_trainable) for module in modules)


def log_trainable_parameters(model):
    print(f"Model: {num_parameters(model) - num_parameters(model.roberta.embeddings):,} parameters")
    print \
        (f"Model: {num_parameters(model, only_trainable=True) - num_parameters(model.roberta.embeddings, only_trainable=True):,} trainable parameters")
    print(f"Core transformer (incl. adapters): {num_parameters(model.roberta.encoder):,} parameters")
    print \
        (f"Core transformer (incl. adapters): {num_parameters(model.roberta.encoder, only_trainable=True):,} trainable parameters")
    char_components = [
        model.roberta.char_embeddings,
        model.roberta.initial_char_encoder,
        model.roberta.chars_to_molecules,
        model.roberta.final_char_encoder,
        model.roberta.projection,
    ]
    print(f"Char components: {num_parameters(*char_components):,} parameters")
    print(f"Char components: {num_parameters(*char_components, only_trainable=True):,} trainable parameters")
    if hasattr(model, "lm_head"):
        print(f"LM head: {num_parameters(model.lm_head.dense, model.lm_head.layer_norm):,} parameters")
        print(f"LM head: {num_parameters(model.lm_head.dense, model.lm_head.layer_norm, only_trainable=True):,} trainable parameters")
    print(f"Subword embeddings: {num_parameters(model.roberta.embeddings):,} parameters")
    print(f"Subword embeddings: {num_parameters(model.roberta.embeddings, only_trainable=True):,} trainable parameters")
