"""Shared semantic definitions for reference frames."""

from __future__ import annotations

import torch
import torch.nn.functional as F


FRAME_TYPES = ["camera", "person", "object", "world"]
FRAME_TYPE_TO_ID = {name: idx for idx, name in enumerate(FRAME_TYPES)}
FRAME_ID_TO_TYPE = {idx: name for name, idx in FRAME_TYPE_TO_ID.items()}

# Keep these prompts synchronized with training/eval text-instruction baselines.
FRAME_TEXT_PROMPTS = {
    "camera": "Answer the following spatial question from the camera's perspective (i.e., left/right refers to the viewer's left/right as seen in the image).",
    "person": "Answer the following spatial question from the person's perspective in the scene (i.e., left/right refers to the person's own left/right, which may be opposite to the viewer's).",
    "object": "Answer the following spatial question relative to the specified reference object's orientation.",
    "world": "Answer the following spatial question using absolute/world coordinates.",
}


def build_frame_anchor_vectors(tokenizer, input_embeddings) -> torch.Tensor:
    """Build one semantic anchor vector per frame from natural-language prompts.

    Each anchor is the mean token embedding of the corresponding frame
    instruction prompt in the base language embedding space.
    """
    anchors = []
    emb_device = input_embeddings.weight.device
    with torch.no_grad():
        for frame_type in FRAME_TYPES:
            tokenized = tokenizer(
                FRAME_TEXT_PROMPTS[frame_type],
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(emb_device)
            prompt_embeds = input_embeddings(input_ids).squeeze(0)
            anchor = prompt_embeds.mean(dim=0).to(torch.float32)
            anchors.append(anchor)
    return torch.stack(anchors, dim=0)


def resize_anchor_vectors(anchor_vectors: torch.Tensor, out_features: int) -> torch.Tensor:
    """Project anchor vectors to a target width with parameter-free interpolation."""
    anchor_vectors = anchor_vectors.to(torch.float32)
    if anchor_vectors.shape[-1] == out_features:
        resized = anchor_vectors
    else:
        resized = F.interpolate(
            anchor_vectors.unsqueeze(1),
            size=out_features,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    denom = resized.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return resized / denom


def initialize_frame_token_embeddings(
    input_embeddings,
    token_ids,
    anchor_vectors: torch.Tensor,
) -> None:
    """Initialise learned <frame_*> token embeddings from semantic anchors."""
    if not token_ids:
        return
    weight = input_embeddings.weight
    target = anchor_vectors.to(device=weight.device, dtype=weight.dtype)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    mean_norm = weight.norm(dim=-1).mean().clamp_min(1e-6)
    target = target * mean_norm
    with torch.no_grad():
        for idx, token_id in enumerate(token_ids):
            if token_id is None or token_id < 0:
                continue
            weight[token_id].copy_(target[idx])
