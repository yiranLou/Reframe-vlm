"""
ReFrame-VLM: Complete model assembly.

Qwen2.5-VL + LoRA + Frame Tokens + Relation Head

Two implementation strategies supported:
1. Special Token approach (recommended, works with LLaMA-Factory):
   - Add <frame_camera>, <frame_person>, <frame_object>, <frame_world>
     as special tokens to vocabulary
   - These tokens' embeddings are randomly initialized and trained
   - Prepended to question during data preprocessing
   - Distinct from text prompts: learned continuous vectors

2. Prefix Embedding approach (more principled but harder to integrate):
   - FrameTokenModule generates prefix embeddings
   - Concatenated to input embeddings before transformer
   - Requires custom forward pass
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

from .frame_embedding import FrameEmbedding, FrameTokenModule
from .relation_head import RelationHead, FrameCanonicalProjection, TOTAL_RELATION_DIM


# ── Special Token Constants ──

FRAME_SPECIAL_TOKENS = [
    "<frame_camera>",
    "<frame_person>",
    "<frame_object>",
    "<frame_world>",
]

FRAME_TOKEN_TO_TYPE = {
    "<frame_camera>": "camera",
    "<frame_person>": "person",
    "<frame_object>": "object",
    "<frame_world>": "world",
}

FRAME_TYPE_TO_TOKEN = {v: k for k, v in FRAME_TOKEN_TO_TYPE.items()}


def add_frame_tokens_to_tokenizer(processor):
    """
    Add frame special tokens to tokenizer.
    Returns the number of new tokens added.
    """
    tokenizer = processor.tokenizer
    special_tokens = {"additional_special_tokens": FRAME_SPECIAL_TOKENS}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} frame special tokens to tokenizer")
    return num_added


def get_default_lora_config(rank=64, alpha=128, dropout=0.05):
    """Default LoRA configuration for Qwen2.5-VL."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


class ReFrameVLM(nn.Module):
    """
    Full ReFrame-VLM model.

    Components:
    1. Qwen2.5-VL backbone with LoRA
    2. Frame special tokens (added to vocabulary)
    3. Relation projection head (for consistency loss)
    4. Frame canonical projections (for soft consistency)

    Training flow:
    1. Processor handles images + text (with frame token prepended)
    2. Model forward produces logits (for L_qa) and hidden states
    3. Last-token hidden state -> relation_head -> relation logits
    4. Paired samples' relation logits -> canonical projection -> L_consistency
    """

    def __init__(
        self,
        model_path,
        lora_config=None,
        use_frame_tokens=True,
        use_relation_head=True,
        canonical_dim=64,
    ):
        super().__init__()

        # Load base model
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Qwen2.5-VL has nested text_config; fall back to direct hidden_size
        cfg = self.base_model.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.hidden_dim = cfg.text_config.hidden_size
        else:
            self.hidden_dim = cfg.hidden_size
        self.use_frame_tokens = use_frame_tokens
        self.use_relation_head = use_relation_head

        # Add frame special tokens
        if use_frame_tokens:
            self.processor = AutoProcessor.from_pretrained(model_path)
            num_added = add_frame_tokens_to_tokenizer(self.processor)
            if num_added > 0:
                self.base_model.resize_token_embeddings(
                    len(self.processor.tokenizer)
                )

        # Apply LoRA
        if lora_config is None:
            lora_config = get_default_lora_config()
        self.base_model = get_peft_model(self.base_model, lora_config)

        # Relation head for consistency loss.
        # Cast to bf16 so the dtype matches the base model's hidden states
        # (avoids a runtime error on the first matmul).
        if use_relation_head:
            self.relation_head = RelationHead(
                hidden_dim=self.hidden_dim,
                relation_dim=TOTAL_RELATION_DIM,
            ).to(dtype=torch.bfloat16)
            self.canonical_proj = FrameCanonicalProjection(
                relation_dim=TOTAL_RELATION_DIM,
                canonical_dim=canonical_dim,
            ).to(dtype=torch.bfloat16)
        else:
            self.relation_head = None
            self.canonical_proj = None

        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Trainable ratio: {trainable / total:.2%}")

    def get_relation_logits(self, hidden_states, attention_mask=None):
        """
        Extract relation logits from last non-padding token's hidden state.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) optional
        Returns:
            (batch, relation_dim)
        """
        if self.relation_head is None:
            return None

        if attention_mask is not None:
            # Get last non-padding position
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_indices = torch.arange(
                hidden_states.shape[0], device=hidden_states.device
            )
            last_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]

        # Accelerator / Trainer may reset non-base modules to fp32 after
        # Accelerator.prepare(). Align the input dtype at call time so the
        # matmul inside RelationHead always sees matching dtypes.
        rh_dtype = next(self.relation_head.parameters()).dtype
        if last_hidden.dtype != rh_dtype:
            last_hidden = last_hidden.to(rh_dtype)
        return self.relation_head(last_hidden)

    def get_canonical_projection(self, relation_logits, frame_type_ids):
        """Project relation logits to canonical space."""
        if self.canonical_proj is None:
            return None
        return self.canonical_proj(relation_logits, frame_type_ids)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        labels=None,
        frame_type_ids=None,
        output_hidden_states=False,
        **kwargs,
    ):
        """
        Forward pass.

        When output_hidden_states=True, also computes relation logits
        for consistency loss.
        """
        need_hidden = output_hidden_states or self.use_relation_head

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            output_hidden_states=need_hidden,
            **kwargs,
        )

        result = {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

        if need_hidden and outputs.hidden_states is not None:
            last_hidden_states = outputs.hidden_states[-1]
            result["relation_logits"] = self.get_relation_logits(
                last_hidden_states, attention_mask
            )
        else:
            result["relation_logits"] = None

        if output_hidden_states:
            result["hidden_states"] = outputs.hidden_states

        return result

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Proxy to base model for HF Trainer compatibility."""
        return self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        """Proxy to base model."""
        return self.base_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        """Proxy to base model. Required for gradient checkpointing + LoRA."""
        return self.base_model.enable_input_require_grads()

    def save_pretrained(self, save_dir, **kwargs):
        """Save the LoRA adapter (via PEFT) plus auxiliary modules."""
        # Save PEFT adapter
        self.base_model.save_pretrained(save_dir, **kwargs)
        # Save auxiliary modules
        self.save_auxiliary_modules(save_dir)

    def save_auxiliary_modules(self, save_dir):
        """Save non-LoRA trainable modules (relation head, projections)."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        if self.relation_head is not None:
            torch.save(
                self.relation_head.state_dict(),
                os.path.join(save_dir, "relation_head.pt"),
            )
        if self.canonical_proj is not None:
            torch.save(
                self.canonical_proj.state_dict(),
                os.path.join(save_dir, "canonical_proj.pt"),
            )

    def load_auxiliary_modules(self, save_dir):
        """Load non-LoRA trainable modules."""
        import os

        rh_path = os.path.join(save_dir, "relation_head.pt")
        if os.path.exists(rh_path) and self.relation_head is not None:
            self.relation_head.load_state_dict(torch.load(rh_path))

        cp_path = os.path.join(save_dir, "canonical_proj.pt")
        if os.path.exists(cp_path) and self.canonical_proj is not None:
            self.canonical_proj.load_state_dict(torch.load(cp_path))


def build_model(
    model_path,
    lora_rank=64,
    lora_alpha=128,
    lora_dropout=0.05,
    use_frame_tokens=True,
    use_relation_head=True,
    canonical_dim=64,
):
    """Convenience function to build ReFrameVLM."""
    lora_config = get_default_lora_config(
        rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
    )
    model = ReFrameVLM(
        model_path=model_path,
        lora_config=lora_config,
        use_frame_tokens=use_frame_tokens,
        use_relation_head=use_relation_head,
        canonical_dim=canonical_dim,
    )
    return model
