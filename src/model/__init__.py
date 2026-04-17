from .frame_embedding import FrameEmbedding, FrameTokenModule
from .frame_lora import (
    FrameGateEmbedding,
    SemanticResidualFrameGate,
    patch_lora_with_frame_gating,
    set_frame_type_ids_for_lora,
    save_frame_gates,
    load_frame_gates,
    num_gate_parameters,
)
from .frame_semantics import FRAME_TEXT_PROMPTS, FRAME_TYPE_TO_ID, FRAME_TYPES
from .relation_head import RelationHead
from .reframe_model import ReFrameVLM
