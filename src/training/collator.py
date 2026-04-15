"""
Data collator for ReFrame-VLM training.

Handles:
1. Building Qwen2.5-VL conversation format with frame tokens
2. Processing multi-image inputs
3. Separating single samples and paired samples
4. Creating labels for causal LM training (masking non-answer tokens)

Note: Qwen2.5-VL uses a message-based format where images are specified
as {"type": "image", "image": "file://path"} in the content list.
"""

import torch
from qwen_vl_utils import process_vision_info
from src.model.reframe_model import FRAME_TYPE_TO_TOKEN


# Natural-language frame instructions for the LoRA+text-instruction baseline.
# Kept verbatim identical to the inference-time FRAME_PROMPTS in
# src/eval/run_benchmark.py so that training and eval prompts match exactly.
FRAME_TEXT_PROMPTS = {
    "camera": "Answer the following spatial question from the camera's perspective (i.e., left/right refers to the viewer's left/right as seen in the image).",
    "person": "Answer the following spatial question from the person's perspective in the scene (i.e., left/right refers to the person's own left/right, which may be opposite to the viewer's).",
    "object": "Answer the following spatial question relative to the specified reference object's orientation.",
    "world":  "Answer the following spatial question using absolute/world coordinates.",
}


# Qwen2.5-VL 的 assistant 起止标记
# 实际 token 取决于 chat template，以下为常见格式
ASSISTANT_START_TOKENS = ["<|im_start|>assistant", "<|assistant|>"]
ASSISTANT_END_TOKENS = ["<|im_end|>", "<|endoftext|>"]


class ReFrameCollator:
    """
    Collator that builds model inputs from dataset items.

    For each sample:
    1. Build conversation messages with frame token prepended
    2. Process through Qwen2.5-VL processor
    3. Create labels: only compute loss on assistant response tokens
    """

    def __init__(self, processor, max_length=2048, use_frame_tokens=True,
                 use_frame_text_prompt=False):
        self.processor = processor
        self.max_length = max_length
        self.use_frame_tokens = use_frame_tokens
        # Mutually-exclusive fair-baseline mode: prepend a natural-language
        # frame instruction instead of a learned <frame_*> special token.
        self.use_frame_text_prompt = use_frame_text_prompt
        assert not (use_frame_tokens and use_frame_text_prompt), \
            "use_frame_tokens and use_frame_text_prompt are mutually exclusive"

        # 缓存 assistant 标记的 token ids 用于标签遮罩
        self._cache_special_token_ids()

    def _cache_special_token_ids(self):
        """缓存 chat template 中的特殊 token id，用于定位 assistant 回复的边界。"""
        tokenizer = self.processor.tokenizer

        self.assistant_start_ids = []
        self.assistant_end_ids = []

        for tok in ASSISTANT_START_TOKENS:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                self.assistant_start_ids.append(ids)

        for tok in ASSISTANT_END_TOKENS:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                self.assistant_end_ids.append(ids)

    def _build_question(self, question, choices, frame_type):
        """Build question string with optional frame conditioning and choices."""
        parts = []

        # Prepend learned frame token (Frame LoRA / Full Method)
        if self.use_frame_tokens and frame_type:
            frame_token = FRAME_TYPE_TO_TOKEN.get(frame_type, "<frame_camera>")
            parts.append(frame_token)

        # OR prepend natural-language frame instruction (fair text baseline)
        if self.use_frame_text_prompt and frame_type:
            instr = FRAME_TEXT_PROMPTS.get(frame_type, "")
            if instr:
                parts.append(instr)

        # Question
        parts.append(question)

        # Choices
        if choices:
            opts = ", ".join(str(c) for c in choices)
            parts.append(f"\nOptions: {opts}\nAnswer:")
        else:
            parts.append("\nAnswer:")

        return "\n".join(parts)

    def _build_messages(self, images, question_text, answer):
        """Build Qwen2.5-VL conversation format."""
        content = []
        for img_path in images:
            content.append({"type": "image", "image": f"file://{img_path}"})
        content.append({"type": "text", "text": question_text})

        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        return messages

    def _process_single(self, item):
        """Process a single (non-pair) sample."""
        question_text = self._build_question(
            item["question"], item.get("choices"), item.get("frame_type")
        )
        messages = self._build_messages(
            item["images"], question_text, item["answer"]
        )
        return messages, item.get("frame_type_id", 0)

    def __call__(self, batch):
        """
        Collate a batch of samples.

        Returns dict with:
        - input_ids, attention_mask, pixel_values, image_grid_thw (模型输入)
        - labels: (batch_size, seq_len)，非 assistant 回复位置为 -100
        - frame_type_ids: (batch_size,) tensor
        - pair_indices: list of (idx_a, idx_b) tuples for consistency loss
        """
        all_messages = []
        all_frame_ids = []
        pair_indices = []

        for item in batch:
            if not item["is_pair"]:
                msgs, fid = self._process_single(item)
                all_messages.append(msgs)
                all_frame_ids.append(fid)
            else:
                # Paired sample: expand into two entries
                idx_a = len(all_messages)

                q_a = self._build_question(
                    item["question_a"], item.get("choices_a"),
                    item.get("frame_type_a")
                )
                msgs_a = self._build_messages(
                    item["images_a"], q_a, item["answer_a"]
                )
                all_messages.append(msgs_a)
                all_frame_ids.append(item.get("frame_type_id_a", 0))

                idx_b = len(all_messages)

                q_b = self._build_question(
                    item["question_b"], item.get("choices_b"),
                    item.get("frame_type_b")
                )
                msgs_b = self._build_messages(
                    item["images_b"], q_b, item["answer_b"]
                )
                all_messages.append(msgs_b)
                all_frame_ids.append(item.get("frame_type_id_b", 0))

                pair_indices.append((idx_a, idx_b))

        batch_inputs = self._batch_process(all_messages)
        batch_inputs["frame_type_ids"] = torch.tensor(
            all_frame_ids, dtype=torch.long
        )
        batch_inputs["pair_indices"] = pair_indices

        return batch_inputs

    def _find_subsequence(self, seq, subseq):
        """在 token 序列中查找子序列，返回所有起始位置。"""
        positions = []
        n = len(seq)
        m = len(subseq)
        for i in range(n - m + 1):
            if seq[i:i + m] == subseq:
                positions.append(i)
        return positions

    def _mask_non_assistant_tokens(self, input_ids, labels):
        """
        将非 assistant 回复部分的 label 设为 -100。

        策略：找到 assistant 起止标记之间的 token，只保留这些 token 的 label。
        对 Qwen2.5-VL，chat template 格式大致为：
          <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>

        我们只对 assistant 部分计算 loss。
        """
        batch_size, seq_len = input_ids.shape

        for b in range(batch_size):
            ids = input_ids[b].tolist()

            # 找到所有 assistant 起始位置
            start_positions = []
            for start_ids in self.assistant_start_ids:
                positions = self._find_subsequence(ids, start_ids)
                for pos in positions:
                    # assistant 回复从标记之后开始
                    start_positions.append(pos + len(start_ids))

            # 找到所有 assistant 结束位置
            end_positions = []
            for end_ids in self.assistant_end_ids:
                positions = self._find_subsequence(ids, end_ids)
                end_positions.extend(positions)

            if not start_positions:
                # 找不到 assistant 标记，回退：只遮罩前半部分
                # 简单启发式：最后 1/3 是 assistant 回复
                cutoff = seq_len * 2 // 3
                labels[b, :cutoff] = -100
                continue

            # 先全部遮罩
            labels[b, :] = -100

            # 只暴露 assistant 回复区间
            start_positions.sort()
            end_positions.sort()

            for start in start_positions:
                # 找到此 start 之后最近的 end
                end = seq_len  # 默认到序列末尾
                for ep in end_positions:
                    if ep > start:
                        end = ep
                        break
                # 暴露 [start, end) 区间的 labels
                labels[b, start:end] = input_ids[b, start:end]

        return labels

    def _batch_process(self, all_messages):
        """
        Process messages through Qwen2.5-VL processor.
        构造完整的模型输入，包括正确的标签遮罩。
        """
        all_texts = []
        all_image_list = []

        for messages in all_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            all_texts.append(text)

            image_inputs, _ = process_vision_info(messages)
            all_image_list.append(image_inputs)

        # 展平图片列表（processor 期望一个平坦的图片列表）
        flat_images = []
        for imgs in all_image_list:
            if imgs:
                if isinstance(imgs, list):
                    flat_images.extend(imgs)
                else:
                    flat_images.append(imgs)

        inputs = self.processor(
            text=all_texts,
            images=flat_images if flat_images else None,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        # 创建标签并遮罩非 assistant 部分
        labels = inputs["input_ids"].clone()

        # 遮罩 padding
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        # 遮罩非 assistant 回复的 token
        labels = self._mask_non_assistant_tokens(inputs["input_ids"], labels)

        inputs["labels"] = labels

        return inputs
