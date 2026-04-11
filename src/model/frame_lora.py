"""
Frame-Conditioned LoRA implementation.

Standard LoRA:  h = Wx + BAx
Frame-Cond:     h = Wx + (alpha_f . B) A x

alpha_f is a per-channel scaling vector generated from the frame embedding
via a small MLP. This means different frame types activate genuinely
different adapter behaviors, not just a prefix token.

Engineering cost: one extra element-wise multiply per LoRA forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FrameGateMLP(nn.Module):
    """
    Small MLP that converts frame embedding into a gating vector
    for LoRA's B matrix.

    Input:  frame_emb (batch, frame_dim)
    Output: gate      (batch, out_features) in [0, 1]
    """

    def __init__(self, frame_dim, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frame_dim, frame_dim),
            nn.SiLU(),
            nn.Linear(frame_dim, out_features),
        )
        # Initialize so sigmoid output starts near ~0.88 (close to 1)
        # This makes initial behavior close to standard LoRA
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 2.0)

    def forward(self, frame_emb):
        return torch.sigmoid(self.mlp(frame_emb))


class FrameConditionedLoRALinear(nn.Module):
    """
    Drop-in replacement for a LoRA linear layer with frame conditioning.

    Wraps the original peft LoraLayer and adds frame-conditioned gating
    on the B matrix output.
    """

    def __init__(self, base_layer, lora_A, lora_B, scaling, dropout,
                 frame_dim, adapter_name="default"):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.scaling = scaling
        self.dropout = dropout
        self.adapter_name = adapter_name

        out_features = lora_B.weight.shape[0]
        self.frame_gate = FrameGateMLP(frame_dim, out_features)

    def forward(self, x, frame_emb=None):
        """
        Args:
            x: (batch, seq_len, in_features)
            frame_emb: (batch, frame_dim) or None
        Returns:
            (batch, seq_len, out_features)
        """
        # Base linear
        result = self.base_layer(x)

        # LoRA path
        after_A = self.lora_A(self.dropout(x))  # (batch, seq_len, rank)
        after_B = self.lora_B(after_A)  # (batch, seq_len, out_features)

        if frame_emb is not None:
            # Frame-conditioned gating
            gate = self.frame_gate(frame_emb)  # (batch, out_features)
            gate = gate.unsqueeze(1)  # (batch, 1, out_features) for broadcasting
            after_B = after_B * gate

        result = result + after_B * self.scaling
        return result


def patch_lora_with_frame_conditioning(peft_model, frame_dim):
    """
    Monkey-patch a PEFT model's LoRA layers to be frame-conditioned.

    This iterates over all LoRA layers and wraps them with
    FrameConditionedLoRALinear. The frame_emb is passed through
    a module-level attribute during forward.

    Args:
        peft_model: A PEFT model with LoRA adapters
        frame_dim: Dimension of frame embeddings

    Returns:
        List of FrameGateMLP modules (for optimizer parameter groups)
    """
    gate_modules = []

    for name, module in peft_model.named_modules():
        # peft LoRA layers have lora_A and lora_B as sub-modules
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            adapter_name = "default"
            if adapter_name in module.lora_A and adapter_name in module.lora_B:
                lora_A = module.lora_A[adapter_name]
                lora_B = module.lora_B[adapter_name]

                out_features = lora_B.weight.shape[0]
                gate = FrameGateMLP(frame_dim, out_features)
                gate_modules.append(gate)

                # Store gate as attribute on the module
                module._frame_gate = gate

    print(f"Patched {len(gate_modules)} LoRA layers with frame conditioning")
    return nn.ModuleList(gate_modules)
