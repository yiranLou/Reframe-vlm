"""
Unified training dataset for ReFrame-VLM.

Supports two modes:
1. QA mode: standard single-sample training
2. Consistency mode: paired cross-frame samples
3. Both mode: mixed batches

Frame type IDs:
  0 = camera, 1 = person, 2 = object, 3 = world
"""

import json
import random
from torch.utils.data import Dataset


FRAME_TYPE_MAP = {
    "camera": 0,
    "person": 1,
    "object": 2,
    "world": 3,
}


class ReFrameDataset(Dataset):
    """
    Main training dataset.

    Args:
        data_path: Path to train.jsonl (unified format)
        consistency_pairs_path: Path to consistency_pairs.jsonl
        mode: "qa" | "consistency" | "both"
        view_permutation: Whether to randomly shuffle view order
        view_permutation_prob: Probability of shuffling (default 0.5)
    """

    def __init__(
        self,
        data_path,
        consistency_pairs_path=None,
        mode="qa",
        view_permutation=False,
        view_permutation_prob=0.5,
    ):
        self.mode = mode
        self.view_permutation = view_permutation
        self.view_permutation_prob = view_permutation_prob
        self.samples = []
        self.pairs = []

        # Load main data
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        # Load consistency pairs
        if consistency_pairs_path and mode in ("consistency", "both"):
            with open(consistency_pairs_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.pairs.append(json.loads(line))

        print(
            f"ReFrameDataset loaded: {len(self.samples)} samples, "
            f"{len(self.pairs)} pairs, mode={mode}, "
            f"view_permutation={view_permutation}"
        )

    def __len__(self):
        if self.mode == "consistency":
            return len(self.pairs)
        elif self.mode == "both":
            return len(self.samples) + len(self.pairs)
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "consistency":
            return self._get_pair(idx)
        elif self.mode == "both":
            if idx < len(self.samples):
                return self._get_single(idx)
            else:
                return self._get_pair(idx - len(self.samples))
        else:
            return self._get_single(idx)

    def _maybe_permute_views(self, images):
        """Randomly shuffle view order for augmentation."""
        if (
            self.view_permutation
            and len(images) > 1
            and random.random() < self.view_permutation_prob
        ):
            images = images.copy()
            random.shuffle(images)
        return images

    def _get_single(self, idx):
        sample = self.samples[idx]
        images = sample["images"]
        images = self._maybe_permute_views(images)

        return {
            "id": sample["id"],
            "images": images,
            "question": sample["question"],
            "choices": sample.get("choices"),
            "answer": sample["answer"],
            "frame_type": sample["frame_type"],
            "frame_type_id": FRAME_TYPE_MAP.get(sample["frame_type"], 0),
            "is_pair": False,
        }

    def _get_pair(self, idx):
        pair = self.pairs[idx]
        sa = pair["sample_a"]
        sb = pair["sample_b"]

        images_a = self._maybe_permute_views(sa["images"])
        images_b = self._maybe_permute_views(sb["images"])

        return {
            "id": pair["pair_id"],
            "is_pair": True,
            # Sample A
            "images_a": images_a,
            "question_a": sa["question"],
            "choices_a": sa.get("choices"),
            "answer_a": sa["answer"],
            "frame_type_a": sa["frame_type"],
            "frame_type_id_a": FRAME_TYPE_MAP.get(sa["frame_type"], 0),
            # Sample B
            "images_b": images_b,
            "question_b": sb["question"],
            "choices_b": sb.get("choices"),
            "answer_b": sb["answer"],
            "frame_type_b": sb["frame_type"],
            "frame_type_id_b": FRAME_TYPE_MAP.get(sb["frame_type"], 0),
        }


class ReFrameEvalDataset(Dataset):
    """
    Evaluation dataset. Simpler than training - no pairs, no augmentation.
    """

    def __init__(self, data_path):
        self.samples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        print(f"EvalDataset loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "id": sample["id"],
            "images": sample["images"],
            "question": sample["question"],
            "choices": sample.get("choices"),
            "answer": sample["answer"],
            "frame_type": sample.get("frame_type", "camera"),
            "frame_type_id": FRAME_TYPE_MAP.get(sample.get("frame_type", "camera"), 0),
        }
