"""
Sanity check: confirm that LoRA + text-instruction SFT loss masking only
includes assistant-response tokens.

The text-instruction training reported ``loss ≈ 0.23`` while Frame LoRA hit
``loss ≈ 3.4`` on the same data. Most of that gap is the +4 added vocab rows
(softmax denominator noise) — but we want to *prove* it isn't a label-masking
bug that exposed the long natural-language instruction in the loss target.

What we do:
  1. Build the ReFrameCollator with use_frame_text_prompt=True (P0-d's
     training setting).
  2. Pull a few samples from the training jsonl through it.
  3. For each, find:
       - the index range whose label is NOT -100 (i.e., entered loss),
       - the corresponding decoded text.
  4. Verify the unmasked region equals the assistant answer (and only that).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoProcessor

from src.training.collator import ReFrameCollator
from src.training.dataset import ReFrameDataset


MODEL_PATH = "models/qwen25-vl-7b"
N_SAMPLES = 3


def main():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    collator = ReFrameCollator(
        processor=processor,
        max_length=768,
        use_frame_tokens=False,
        use_frame_text_prompt=True,   # the P0-d setting
    )

    dataset = ReFrameDataset(
        data_path="data/processed/train.jsonl",
        mode="qa",
        view_permutation=False,
    )

    items = [dataset[i] for i in range(N_SAMPLES)]
    batch = collator(items)
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(f"\nBatch shapes: input_ids={tuple(input_ids.shape)}  "
          f"labels={tuple(labels.shape)}")

    tok = processor.tokenizer
    for i in range(N_SAMPLES):
        ids = input_ids[i].tolist()
        lab = labels[i].tolist()
        unmasked_idx = [j for j, v in enumerate(lab) if v != -100]
        unmasked_tok = [ids[j] for j in unmasked_idx]
        unmasked_text = tok.decode(unmasked_tok, skip_special_tokens=False)
        full_text = tok.decode(ids, skip_special_tokens=False)
        gt_answer = items[i]["answer"]
        frame = items[i].get("frame_type")

        print("\n" + "=" * 78)
        print(f"sample {i}  frame_type={frame}  answer={gt_answer!r}")
        print(f"  total tokens     = {len(ids)}")
        print(f"  unmasked tokens  = {len(unmasked_idx)}")
        print(f"  unmasked range   = {unmasked_idx[0] if unmasked_idx else '-'}"
              f"..{unmasked_idx[-1] if unmasked_idx else '-'}")
        print(f"  unmasked decoded = {unmasked_text!r}")
        # Visually show the boundary
        if unmasked_idx:
            head = tok.decode(ids[max(0, unmasked_idx[0] - 8):unmasked_idx[0]],
                              skip_special_tokens=False)
            tail = tok.decode(ids[unmasked_idx[-1] + 1:
                                  unmasked_idx[-1] + 9],
                              skip_special_tokens=False)
            print(f"  context BEFORE  = {head!r}")
            print(f"  context AFTER   = {tail!r}")

        # Cheap check: unmasked decoded should contain the gt_answer.
        ok = gt_answer.strip().lower() in unmasked_text.lower()
        print(f"  GT answer found in unmasked region? {ok}")

        # Also confirm we did NOT include the long text instruction in the
        # loss target (catches the suspected bug).
        bad_phrases = (
            "Answer the following spatial question",
            "from the camera's perspective",
            "from the person's perspective",
        )
        bleed = [p for p in bad_phrases if p.lower() in unmasked_text.lower()]
        if bleed:
            print(f"  ⚠  INSTRUCTION BLED INTO LOSS: {bleed}")
        else:
            print(f"  ✓  no frame-instruction text in loss target")


if __name__ == "__main__":
    main()
