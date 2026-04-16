"""
Loss functions for ReFrame-VLM.

L_total = L_qa + lambda_consist * L_consistency

L_qa: cross-entropy from the LM head (computed by model).
L_consistency: MSE between paired samples in a frame-canonical space.

The canonical projection module lives on the model (ReFrameVLM.canonical_proj)
so that it is automatically moved to GPU, added to the optimizer, and saved
with the checkpoint. ReFrameLoss itself is stateless.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReFrameLoss(nn.Module):
    """
    Stateless combined loss. Takes the canonical projection module as
    an external argument in forward() so projections live on the model.
    """

    def __init__(self, lambda_consistency=0.1):
        super().__init__()
        self.lambda_consistency = lambda_consistency

    def forward(
        self,
        qa_loss,
        relation_logits=None,
        pair_indices=None,
        frame_type_ids=None,
        canonical_proj=None,
    ):
        """
        Args:
            qa_loss: scalar CE loss from model.
            relation_logits: (batch, relation_dim) from RelationHead.
            pair_indices: list of (idx_a, idx_b) tuples into the flat batch.
            frame_type_ids: (batch,) long tensor of frame ids.
            canonical_proj: FrameCanonicalProjection module (on model).
        Returns:
            dict(total_loss, qa_loss, consistency_loss)
        """
        zero = torch.tensor(0.0, device=qa_loss.device, dtype=qa_loss.dtype)
        result = {
            "total_loss": qa_loss,
            "qa_loss": qa_loss.detach(),
            "consistency_loss": zero,
            "consistency_loss_raw": zero,
        }

        active = (
            self.lambda_consistency > 0
            and relation_logits is not None
            and pair_indices
            and frame_type_ids is not None
            and canonical_proj is not None
        )
        if not active:
            return result

        device = relation_logits.device
        idx_a = torch.tensor([p[0] for p in pair_indices], device=device,
                             dtype=torch.long)
        idx_b = torch.tensor([p[1] for p in pair_indices], device=device,
                             dtype=torch.long)

        r_a = relation_logits[idx_a]
        r_b = relation_logits[idx_b]
        ft_a = frame_type_ids.to(device)[idx_a]
        ft_b = frame_type_ids.to(device)[idx_b]

        proj_a = canonical_proj(r_a, ft_a)
        proj_b = canonical_proj(r_b, ft_b)

        l_consist = F.mse_loss(proj_a.float(), proj_b.float())
        result["consistency_loss_raw"] = l_consist
        result["consistency_loss"] = l_consist.detach()
        result["total_loss"] = qa_loss + self.lambda_consistency * l_consist
        return result
