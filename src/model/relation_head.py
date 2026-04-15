"""
Relation Projection Head for Frame Consistency Loss.

Takes the last-token hidden state from the LLM and projects it to a
14-dim relation logit vector:
  - 8 horizontal directions: N/NE/E/SE/S/SW/W/NW
  - 3 vertical relations: above/below/level
  - 3 distance relations: near/mid/far

For consistency loss, each frame type has its own projection to a
shared canonical space.
"""

import torch
import torch.nn as nn


RELATION_DIMS = {
    "direction_8": 8,   # N, NE, E, SE, S, SW, W, NW
    "vertical_3": 3,    # above, below, level
    "distance_3": 3,    # near, mid, far
}
TOTAL_RELATION_DIM = sum(RELATION_DIMS.values())  # 14


class RelationHead(nn.Module):
    """
    Projects LLM hidden state to relation logits.

    Input:  hidden_state from last token, (batch, hidden_dim)
    Output: relation logits, (batch, 14)
    """

    def __init__(self, hidden_dim, relation_dim=TOTAL_RELATION_DIM):
        super().__init__()
        self.relation_dim = relation_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, relation_dim),
        )

    def forward(self, hidden_state):
        """
        Args:
            hidden_state: (batch, hidden_dim)
        Returns:
            (batch, relation_dim) relation logits
        """
        return self.proj(hidden_state)

    def get_direction_logits(self, logits):
        """Extract 8-direction logits."""
        return logits[:, :8]

    def get_vertical_logits(self, logits):
        """Extract vertical relation logits."""
        return logits[:, 8:11]

    def get_distance_logits(self, logits):
        """Extract distance relation logits."""
        return logits[:, 11:14]


class FrameCanonicalProjection(nn.Module):
    """
    Per-frame-type projection to canonical relation space.

    Used for soft consistency loss when exact pose transforms are unavailable.
    Each frame type has a separate linear projection, and consistency is
    enforced in the shared canonical space.
    """

    FRAME_NAMES = ["camera", "person", "object", "world"]

    def __init__(self, relation_dim=TOTAL_RELATION_DIM, canonical_dim=64):
        super().__init__()
        self.canonical_dim = canonical_dim
        self.projections = nn.ModuleDict({
            name: nn.Linear(relation_dim, canonical_dim)
            for name in self.FRAME_NAMES
        })

    def forward(self, relation_logits, frame_type_ids):
        """
        Project relation logits to canonical space based on frame type.

        Args:
            relation_logits: (batch, relation_dim)
            frame_type_ids: (batch,) LongTensor in [0, 3]
        Returns:
            (batch, canonical_dim)
        """
        batch_size = relation_logits.shape[0]
        device = relation_logits.device

        # Match projection dtype to avoid fp32/bf16 mismatch under Accelerator.
        proj_dtype = next(self.projections[self.FRAME_NAMES[0]].parameters()).dtype
        if relation_logits.dtype != proj_dtype:
            relation_logits = relation_logits.to(proj_dtype)

        projected = torch.zeros(
            batch_size, self.canonical_dim,
            device=device, dtype=proj_dtype
        )

        for i, name in enumerate(self.FRAME_NAMES):
            mask = (frame_type_ids == i)
            if mask.any():
                projected[mask] = self.projections[name](relation_logits[mask])

        return projected
