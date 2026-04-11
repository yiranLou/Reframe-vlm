"""
Frame Type Embedding modules.

Two implementations:
1. FrameEmbedding: Simple embedding lookup (used as gate signal)
2. FrameTokenModule: Learnable prefix tokens per frame type
"""

import torch
import torch.nn as nn


class FrameEmbedding(nn.Module):
    """
    4 learnable embeddings (camera/person/object/world).
    Used as gate signal for Frame-Conditioned LoRA.
    Dimension matches LLM hidden_dim.
    """

    NUM_FRAME_TYPES = 4
    FRAME_NAMES = ["camera", "person", "object", "world"]
    FRAME_TO_ID = {name: i for i, name in enumerate(FRAME_NAMES)}

    def __init__(self, hidden_dim, num_frame_types=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_frame_types, hidden_dim)
        # Initialize close to 1 so initial behavior is near standard LoRA
        nn.init.ones_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight.add_(
                torch.randn_like(self.embedding.weight) * 0.02
            )

    def forward(self, frame_type_ids):
        """
        Args:
            frame_type_ids: (batch,) LongTensor with values in [0, 3]
        Returns:
            (batch, hidden_dim) frame embeddings
        """
        return self.embedding(frame_type_ids)

    @staticmethod
    def frame_name_to_id(name):
        return FrameEmbedding.FRAME_TO_ID.get(name.lower(), 0)


class FrameTokenModule(nn.Module):
    """
    Learnable prefix tokens for each frame type.
    Prepended to input embedding sequence.

    Each frame type gets `tokens_per_frame` learnable vectors.
    These are NOT text tokens - they are continuous vectors that
    encode spatial reasoning modes, distinct from text prompts.
    """

    def __init__(self, hidden_dim, num_frame_types=4, tokens_per_frame=4):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.hidden_dim = hidden_dim

        # (num_types, tokens_per_frame, hidden_dim)
        self.frame_tokens = nn.Parameter(
            torch.randn(num_frame_types, tokens_per_frame, hidden_dim) * 0.02
        )

    def forward(self, frame_type_ids):
        """
        Args:
            frame_type_ids: (batch,) LongTensor
        Returns:
            (batch, tokens_per_frame, hidden_dim)
        """
        return self.frame_tokens[frame_type_ids]

    @property
    def num_tokens(self):
        return self.tokens_per_frame
