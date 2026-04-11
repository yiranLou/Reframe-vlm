"""
Custom trainer for ReFrame-VLM.

Extends HuggingFace Trainer to handle:
1. Frame type IDs passed to model
2. Consistency loss from paired samples
3. Mixed batches (single + paired samples)
4. Logging of individual loss components
"""

import torch
from transformers import Trainer
from .losses import ReFrameLoss


class ReFrameTrainer(Trainer):
    """
    Custom trainer that adds consistency loss to the standard HF training loop.

    The model's forward pass returns:
    - loss: standard L_qa (cross-entropy)
    - relation_logits: for consistency loss

    We compute the combined loss here.
    """

    def __init__(self, *args, lambda_consistency=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.reframe_loss = ReFrameLoss(lambda_consistency=lambda_consistency)
        self._consistency_loss_sum = 0.0
        self._consistency_loss_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to add consistency loss.
        """
        # Extract custom fields before passing to model
        frame_type_ids = inputs.pop("frame_type_ids", None)
        pair_indices = inputs.pop("pair_indices", None)

        # Determine if we need relation logits
        need_consistency = (
            pair_indices is not None
            and len(pair_indices) > 0
            and self.reframe_loss.lambda_consistency > 0
        )

        # Forward pass
        if need_consistency:
            inputs["output_hidden_states"] = True

        outputs = model(**inputs, frame_type_ids=frame_type_ids)

        qa_loss = outputs["loss"]
        relation_logits = outputs.get("relation_logits")

        # Compute combined loss
        loss_dict = self.reframe_loss(
            qa_loss=qa_loss,
            relation_logits=relation_logits,
            pair_indices=pair_indices,
            frame_type_ids=frame_type_ids,
        )

        total_loss = loss_dict["total_loss"]

        # Track consistency loss for logging
        if loss_dict["consistency_loss"].item() > 0:
            self._consistency_loss_sum += loss_dict["consistency_loss"].item()
            self._consistency_loss_count += 1

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def log(self, logs):
        """Add consistency loss to logs."""
        if self._consistency_loss_count > 0:
            avg_consist = (
                self._consistency_loss_sum / self._consistency_loss_count
            )
            logs["consistency_loss"] = round(avg_consist, 6)
            self._consistency_loss_sum = 0.0
            self._consistency_loss_count = 0

        super().log(logs)


class BaselineTrainer(Trainer):
    """
    Standard trainer for baseline LoRA fine-tuning.
    No frame tokens, no consistency loss.
    """
    pass
