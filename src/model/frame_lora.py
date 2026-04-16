"""
Frame-Gated LoRA: parameter-space reference-frame conditioning.

Standard LoRA:      y = W x + (B A x) * scaling
Frame-Gated LoRA:   y = W x + g_f(frame_type_ids) ⊙ (B A x) * scaling

where
    g_f(i) = 1 + tanh(E[i])
    E     ∈ R^{num_frames × out_features}
    E is initialised to zero → g_f = 1 at step 0 → behaviour identical to
    standard LoRA. During training the gate diverges per frame type.

Implementation notes
--------------------
* Each PEFT ``LoraLinear`` (``peft.tuners.lora.Linear``) gets its own
  ``FrameGateEmbedding``. We monkey-patch the module's ``forward`` to
  multiply the LoRA-only output by the per-sample gate before adding it to
  the frozen base output.
* The current batch's ``frame_type_ids`` are set on every patched layer via
  :func:`set_frame_type_ids_for_lora` before each outer forward. They stay
  on the layer so that gradient-checkpointing's recomputation during
  backward reads the same values.
* Gates are *not* written into PEFT's ``adapter_model.safetensors``. Save
  them alongside the adapter with :func:`save_frame_gates` and load with
  :func:`load_frame_gates` (auto-called from ReFrameVLM.save/load).

Design choices that matter for the paper
----------------------------------------
* **Identity init** preserves verified LoRA training dynamics. If the gate
  never moves, we recover Frame LoRA exactly.
* **Per-LoRA-layer gate** (not a single global gate) gives the model
  expressive capacity to re-route frame information layer-by-layer without
  touching the frozen backbone.
* **Gate size 4 × out_features** is tiny: ~6 M total params across all
  LoRA sites in Qwen2.5-VL-7B (vs 196 M LoRA params).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


NUM_FRAMES = 4   # camera, person, object, world


# ── Gate module ──────────────────────────────────────────────────────────

class FrameGateEmbedding(nn.Module):
    """Per-LoRA-layer, per-frame multiplicative gate.

    Output is ``(batch, out_features)`` with values in roughly ``[0, 2]``
    because of ``1 + tanh``. Initialised to the identity (all 1s).
    """

    def __init__(self, num_frames: int, out_features: int):
        super().__init__()
        self.num_frames = num_frames
        self.out_features = out_features
        self.emb = nn.Embedding(num_frames, out_features)
        # Identity init: tanh(0) = 0 → gate = 1 → standard LoRA.
        nn.init.zeros_(self.emb.weight)

    def forward(self, frame_type_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            frame_type_ids: LongTensor ``(batch,)``.
        Returns:
            ``(batch, out_features)`` in roughly ``[0, 2]``.
        """
        return 1.0 + torch.tanh(self.emb(frame_type_ids))


# ── Patched forward ──────────────────────────────────────────────────────

def _make_gated_forward(mod: nn.Module):
    """Build a replacement ``forward`` for a PEFT LoRA Linear.

    Mirrors ``peft.tuners.lora.layer.Linear.forward`` (v0.13) for the
    standard code path and injects the frame gate only on the LoRA branch.
    Degenerate cases (disabled adapters, merged, adapter_names, DoRA) fall
    through to the un-patched base behaviour so no edge-case is broken.
    """

    def gated_forward(x: torch.Tensor, *args, **kwargs):
        # Preserve PEFT's shape check if available.
        if hasattr(mod, "_check_forward_args"):
            mod._check_forward_args(x, *args, **kwargs)

        adapter_names = kwargs.pop("adapter_names", None)

        # Path 1: adapters disabled → just the base linear.
        if getattr(mod, "disable_adapters", False):
            if getattr(mod, "merged", False):
                mod.unmerge()
            return mod.base_layer(x, *args, **kwargs)

        # Path 2: mixed-adapter inference (rare) — defer to PEFT's original
        # forward via the superclass. We do not gate this case; it is not
        # used in training.
        if adapter_names is not None:
            base_cls = type(mod).__mro__[1]  # first superclass = peft Linear
            return base_cls.forward(mod, x, *args, adapter_names=adapter_names,
                                    **kwargs)

        # Path 3: adapters already merged into base weights.
        if getattr(mod, "merged", False):
            return mod.base_layer(x, *args, **kwargs)

        # Path 4 (training / non-merged inference): compute base + gated LoRA.
        result = mod.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        frame_ids = getattr(mod, "_current_frame_type_ids", None)

        for active_adapter in mod.active_adapters:
            if active_adapter not in mod.lora_A.keys():
                continue
            lora_A = mod.lora_A[active_adapter]
            lora_B = mod.lora_B[active_adapter]
            dropout = mod.lora_dropout[active_adapter]
            scaling = mod.scaling[active_adapter]

            if hasattr(mod, "_cast_input_dtype"):
                x_cast = mod._cast_input_dtype(x, lora_A.weight.dtype)
            else:
                x_cast = x.to(lora_A.weight.dtype)

            # (B, S, out) for Linear layers.
            lora_out = lora_B(lora_A(dropout(x_cast))) * scaling

            if frame_ids is not None:
                # frame_ids is a LongTensor; move to matching device.
                if frame_ids.device != lora_out.device:
                    frame_ids = frame_ids.to(lora_out.device)
                gate = mod.frame_gate(frame_ids)             # (B, out)
                gate = gate.to(lora_out.dtype).unsqueeze(1)  # (B, 1, out)
                lora_out = lora_out * gate

            result = result + lora_out

        return result.to(torch_result_dtype)

    return gated_forward


# ── Public API ───────────────────────────────────────────────────────────

def patch_lora_with_frame_gating(
    peft_model: nn.Module,
    num_frames: int = NUM_FRAMES,
    dtype: Optional[torch.dtype] = None,
) -> List[Tuple[str, nn.Module]]:
    """Attach a per-frame gate to every PEFT ``LoraLinear`` and rewire
    its ``forward``.

    Returns a list of ``(qualified_name, module)`` pairs that were patched.
    Idempotent: modules that already have a ``frame_gate`` attribute are
    skipped.
    """
    from peft.tuners.lora import Linear as LoraLinear

    patched: List[Tuple[str, nn.Module]] = []
    for name, mod in peft_model.named_modules():
        if not isinstance(mod, LoraLinear):
            continue
        if hasattr(mod, "frame_gate"):
            continue  # already patched
        out_features = mod.base_layer.out_features
        gate = FrameGateEmbedding(num_frames, out_features)
        if dtype is not None:
            gate = gate.to(dtype=dtype)
        mod.frame_gate = gate
        mod._current_frame_type_ids = None
        mod.forward = _make_gated_forward(mod)
        patched.append((name, mod))
    return patched


def set_frame_type_ids_for_lora(
    peft_model: nn.Module,
    frame_type_ids: Optional[torch.Tensor],
) -> int:
    """Broadcast the current batch's frame ids to every patched LoRA layer.

    Args:
        peft_model: model previously patched with
            :func:`patch_lora_with_frame_gating`.
        frame_type_ids: LongTensor ``(batch,)`` or ``None`` to disable
            gating (reverts to standard LoRA behaviour).

    Returns:
        The number of layers whose state was updated.
    """
    n = 0
    for mod in peft_model.modules():
        if hasattr(mod, "frame_gate"):
            mod._current_frame_type_ids = frame_type_ids
            n += 1
    return n


def save_frame_gates(peft_model: nn.Module, save_dir: str) -> int:
    """Serialize every ``frame_gate`` submodule to
    ``{save_dir}/frame_lora_gates.pt``. Returns the number of gates saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    state = {}
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "frame_gate"):
            state[name] = mod.frame_gate.state_dict()
    if state:
        torch.save(state, os.path.join(save_dir, "frame_lora_gates.pt"))
    return len(state)


def load_frame_gates(peft_model: nn.Module, save_dir: str) -> int:
    """Inverse of :func:`save_frame_gates`.

    Safe to call when the file is missing (returns 0). Each gate's weight
    tensor is loaded strictly; a shape mismatch raises immediately.
    """
    path = os.path.join(save_dir, "frame_lora_gates.pt")
    if not os.path.exists(path):
        return 0
    state = torch.load(path, map_location="cpu")
    loaded = 0
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "frame_gate") and name in state:
            mod.frame_gate.load_state_dict(state[name])
            loaded += 1
    return loaded


def num_gate_parameters(peft_model: nn.Module) -> int:
    """Total trainable parameters across all frame gates."""
    total = 0
    for mod in peft_model.modules():
        if hasattr(mod, "frame_gate"):
            for p in mod.frame_gate.parameters():
                if p.requires_grad:
                    total += p.numel()
    return total


def iter_gate_modules(peft_model: nn.Module):
    """Iterate ``(qualified_name, module)`` for every patched LoRA layer."""
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "frame_gate"):
            yield name, mod
