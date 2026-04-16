"""
Custom trainer for ReFrame-VLM.

Extends HuggingFace Trainer to:
1. Pass frame_type_ids to the model forward pass.
2. Compute consistency loss from paired-sample relation logits.
3. Route projection through the model's registered canonical_proj module.
4. Log L_consistency alongside L_qa.
"""

import torch
from transformers import Trainer
from .losses import ReFrameLoss


class ReFrameTrainer(Trainer):
    """
    Trainer that composes L_qa (from model) with L_consistency (pair-wise).

    The canonical projection lives on the model so that:
      - parameters are included in the optimizer automatically
      - tensors are on the right device/dtype
      - weights are saved with the adapter checkpoint
    """

    def __init__(self, *args, lambda_consistency=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.reframe_loss = ReFrameLoss(lambda_consistency=lambda_consistency)
        self._consist_sum = 0.0
        self._consist_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        frame_type_ids = inputs.pop("frame_type_ids", None)
        pair_indices = inputs.pop("pair_indices", None)

        need_consistency = (
            pair_indices is not None
            and len(pair_indices) > 0
            and self.reframe_loss.lambda_consistency > 0
        )
        if need_consistency:
            inputs["output_hidden_states"] = True

        outputs = model(**inputs, frame_type_ids=frame_type_ids)

        qa_loss = outputs["loss"]
        relation_logits = outputs.get("relation_logits")

        # The canonical projection module is a submodule of the ReFrameVLM.
        # Under DDP / Accelerate, unwrap the model before looking it up.
        if hasattr(self, "accelerator"):
            base = self.accelerator.unwrap_model(model)
        else:
            base = model.module if hasattr(model, "module") else model
        canonical_proj = getattr(base, "canonical_proj", None)

        loss_dict = self.reframe_loss(
            qa_loss=qa_loss,
            relation_logits=relation_logits,
            pair_indices=pair_indices,
            frame_type_ids=frame_type_ids,
            canonical_proj=canonical_proj,
        )

        total_loss = loss_dict["total_loss"]
        c_val = loss_dict["consistency_loss"]
        if isinstance(c_val, torch.Tensor) and c_val.item() > 0:
            self._consist_sum += c_val.item()
            self._consist_count += 1

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def log(self, logs, *args, **kwargs):
        if self._consist_count > 0:
            logs["consistency_loss"] = round(
                self._consist_sum / self._consist_count, 6
            )
            self._consist_sum = 0.0
            self._consist_count = 0
        super().log(logs, *args, **kwargs)


class BaselineTrainer(Trainer):
    """Plain Trainer for baseline / frame-only modes (no consistency loss)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Drop any auxiliary fields our collator may include but the base
        # model / PEFT wrapper doesn't understand.
        inputs.pop("pair_indices", None)
        frame_type_ids = inputs.pop("frame_type_ids", None)

        # If the model is ReFrameVLM (frame mode), it accepts frame_type_ids
        # but currently doesn't use them in forward. Pass through when supported.
        try:
            outputs = model(**inputs, frame_type_ids=frame_type_ids)
        except TypeError as exc:
            # Plain HF/PEFT models do not accept frame_type_ids. Re-raise
            # other TypeErrors so real forward bugs are not hidden.
            if "frame_type_ids" not in str(exc):
                raise
            outputs = model(**inputs)

        # ReFrameVLM returns a dict; PEFT-wrapped HF model returns ModelOutput.
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss
