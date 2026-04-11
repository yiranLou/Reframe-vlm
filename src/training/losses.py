"""
Loss functions for ReFrame-VLM.

L_total = L_qa + lambda_consist * L_consistency

Components:
1. L_qa: Standard cross-entropy from the LM head (computed by model)
2. L_consistency: Frame consistency loss between paired samples
   - Soft mode: project to canonical space, enforce MSE
   - Hard mode: use permutation matrix T (requires pose data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameConsistencyLoss(nn.Module):
    """
    Soft frame consistency loss.

    For paired samples (same scene, different frames):
    - Each sample produces relation logits via RelationHead
    - Each frame type has its own projection to canonical space
    - Loss = MSE between canonical projections

    This doesn't require exact pose transforms - just that the
    underlying spatial relationship should be consistent when
    projected through frame-specific transformations.
    """

    def __init__(self, relation_dim=14, canonical_dim=64):
        super().__init__()
        self.canonical_dim = canonical_dim

        # Per-frame-type projections to canonical space
        self.proj_camera = nn.Linear(relation_dim, canonical_dim)
        self.proj_person = nn.Linear(relation_dim, canonical_dim)
        self.proj_object = nn.Linear(relation_dim, canonical_dim)
        self.proj_world = nn.Linear(relation_dim, canonical_dim)

        self._projections = [
            self.proj_camera,
            self.proj_person,
            self.proj_object,
            self.proj_world,
        ]

    def _get_projection(self, frame_type_id):
        """Get projection layer for frame type."""
        return self._projections[frame_type_id]

    def forward(self, relation_logits_a, relation_logits_b,
                frame_type_ids_a, frame_type_ids_b):
        """
        Args:
            relation_logits_a: (N, relation_dim) from sample A
            relation_logits_b: (N, relation_dim) from sample B
            frame_type_ids_a: (N,) frame type IDs for A
            frame_type_ids_b: (N,) frame type IDs for B
        Returns:
            Scalar loss
        """
        if relation_logits_a.shape[0] == 0:
            return torch.tensor(0.0, device=relation_logits_a.device,
                                requires_grad=True)

        # Project each sample to canonical space using its frame's projection
        proj_a = self._batch_project(relation_logits_a, frame_type_ids_a)
        proj_b = self._batch_project(relation_logits_b, frame_type_ids_b)

        # MSE in canonical space
        loss = F.mse_loss(proj_a, proj_b)
        return loss

    def _batch_project(self, relation_logits, frame_type_ids):
        """Project relation logits using per-sample frame projections."""
        batch_size = relation_logits.shape[0]
        device = relation_logits.device
        dtype = relation_logits.dtype

        projected = torch.zeros(
            batch_size, self.canonical_dim,
            device=device, dtype=dtype,
        )

        for ftype_id in range(4):
            mask = (frame_type_ids == ftype_id)
            if mask.any():
                proj = self._projections[ftype_id]
                projected[mask] = proj(relation_logits[mask])

        return projected


class ReFrameLoss(nn.Module):
    """
    Combined loss for ReFrame-VLM training.

    L_total = L_qa + lambda_consistency * L_consistency

    The L_qa comes directly from the model's forward pass (cross-entropy).
    L_consistency is computed here from relation logits of paired samples.
    """

    def __init__(self, lambda_consistency=0.1, relation_dim=14, canonical_dim=64):
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.consistency_loss = FrameConsistencyLoss(
            relation_dim=relation_dim,
            canonical_dim=canonical_dim,
        )

    def forward(
        self,
        qa_loss,
        relation_logits=None,
        pair_indices=None,
        frame_type_ids=None,
    ):
        """
        Args:
            qa_loss: Scalar, cross-entropy loss from model
            relation_logits: (batch, relation_dim), all samples' logits
            pair_indices: list of (idx_a, idx_b) tuples
            frame_type_ids: (batch,) frame type IDs
        Returns:
            Dict with total_loss, qa_loss, consistency_loss
        """
        result = {
            "total_loss": qa_loss,
            "qa_loss": qa_loss.detach(),
            "consistency_loss": torch.tensor(0.0),
        }

        if (
            relation_logits is not None
            and pair_indices
            and frame_type_ids is not None
            and self.lambda_consistency > 0
        ):
            # Extract paired relation logits
            idx_a = torch.tensor(
                [p[0] for p in pair_indices],
                device=relation_logits.device,
            )
            idx_b = torch.tensor(
                [p[1] for p in pair_indices],
                device=relation_logits.device,
            )

            r_a = relation_logits[idx_a]
            r_b = relation_logits[idx_b]
            ft_a = frame_type_ids[idx_a]
            ft_b = frame_type_ids[idx_b]

            l_consist = self.consistency_loss(r_a, r_b, ft_a, ft_b)
            result["consistency_loss"] = l_consist.detach()
            result["total_loss"] = qa_loss + self.lambda_consistency * l_consist

        return result
