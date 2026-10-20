"""G/M/P species-analysis operations for the configurable DT-MAE model."""

import torch


def _check_modalities(model):
    if set(model.encoded_modalities) != {"G", "M", "P"}:
        raise ValueError("Species/module analysis requires encoded modalities G, M, and P.")


def encode_gmp(model, G, M, P):
    """Return embeddings in the model's native latent-coordinate order."""
    _check_modalities(model)
    return model.encoder({"G": G, "M": M, "P": P})[1]


def latent_parts(model, embedding):
    _check_modalities(model)
    widths = [model.latent_dims[m] for m in model.encoded_modalities]
    if embedding.shape[-1] != sum(widths):
        raise ValueError("Embedding width does not match the model's latent dimensions.")
    return dict(zip(model.encoded_modalities, embedding.split(widths, dim=-1)))


def predict_gmp(model, embedding, R):
    parts = latent_parts(model, embedding)
    if set(model.passthrough_modalities) - {"R"}:
        raise ValueError("Species importance supports only R as a passthrough modality.")
    if not {"G", "M", "P"} <= set(model.predictor_output_mods):
        raise ValueError("Species importance requires predictions for G, M, and P.")
    predicted = model.predictor(parts, {"R": R})
    return torch.cat([predicted[m] for m in "GMP"], dim=-1)


def decode_gmp(model, embedding):
    decoded = model.decoder(latent_parts(model, embedding))
    return torch.cat([decoded[m] for m in "GMP"], dim=-1)
