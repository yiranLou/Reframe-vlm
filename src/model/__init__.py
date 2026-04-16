from .frame_embedding import FrameEmbedding, FrameTokenModule
from .frame_lora import (
    FrameGateEmbedding,
    patch_lora_with_frame_gating,
    set_frame_type_ids_for_lora,
    save_frame_gates,
    load_frame_gates,
    num_gate_parameters,
)
from .relation_head import RelationHead
from .reframe_model import ReFrameVLM
