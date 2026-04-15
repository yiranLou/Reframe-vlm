"""
从 ViewSpatial 的 visibility_data 自动生成训练 QA。

生成两类问题：
1. Camera perspective - Relative Direction: 用 2D bbox center 判断空间关系
2. Scene Simulation - Relative Direction: 用 3D 坐标计算 "站在A面向B，C在哪"

每个 camera QA 自动配对一个 scene simulation QA（同场景同物体对），
用于 frame consistency loss。

Usage:
    python data/scripts/generate_training_data.py \
        --data_dir /workspace/datasets/viewspatial \
        --output_dir data/processed \
        --target_size 50000
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from itertools import combinations


# ── Direction helpers ──

DIRECTIONS_4 = ["left", "right", "front", "back"]
DIRECTIONS_8 = [
    "front", "front-right", "right", "back-right",
    "back", "back-left", "left", "front-left",
]
VERTICAL = ["above", "below"]


def angle_to_direction_8(angle_deg):
    """Convert angle (0=front, clockwise) to 8-direction label."""
    # Normalize to [0, 360)
    angle_deg = angle_deg % 360
    # Each sector is 45 degrees, offset by 22.5
    idx = int((angle_deg + 22.5) % 360 / 45)
    return DIRECTIONS_8[idx]


def compute_2d_relation(obj_a, obj_b, img_width=1296, img_height=968):
    """
    Compute spatial relation from camera perspective using 2D bbox centers.
    In image coordinates: x increases rightward, y increases downward.

    Returns direction of B relative to A from camera's viewpoint.
    """
    ax, ay = obj_a["bbox_2d_center"]
    bx, by = obj_b["bbox_2d_center"]

    dx = bx - ax  # positive = right
    dy = ay - by  # flip y: positive = above

    # Filter out objects with 2D centers far outside image bounds
    for x, y in [(ax, ay), (bx, by)]:
        if x < -img_width * 0.5 or x > img_width * 1.5:
            return None
        if y < -img_height * 0.5 or y > img_height * 1.5:
            return None

    # Need some minimum separation to be meaningful
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 30:  # too close in pixels
        return None

    angle = math.degrees(math.atan2(-dx, dy))  # 0=above, 90=left
    # Convert to our convention: 0=front (above in image), clockwise
    angle = (-angle) % 360

    direction = angle_to_direction_8(angle)

    # Simplified to 4 directions for cleaner training signal
    if abs(dx) > abs(dy) * 1.5:
        simple = "right" if dx > 0 else "left"
    elif abs(dy) > abs(dx) * 1.5:
        simple = "above" if dy > 0 else "below"
    else:
        simple = direction  # use 8-direction

    return {
        "direction": simple,
        "direction_8": direction,
        "dx": dx,
        "dy": dy,
        "dist": dist,
    }


def compute_3d_simulation(anchor, facing_target, query_obj):
    """
    Compute: "If you stand at anchor facing facing_target, where is query_obj?"

    Uses 3D coordinates (camera space, but relative positions are valid
    within a single frame).

    Returns direction in the person's reference frame.
    """
    ax, ay, az = anchor["bbox_3d_center"]
    fx, fy, fz = facing_target["bbox_3d_center"]
    qx, qy, qz = query_obj["bbox_3d_center"]

    # Forward direction (in XZ plane, Y is vertical in ScanNet)
    fwd_x = fx - ax
    fwd_z = fz - az
    fwd_len = math.sqrt(fwd_x ** 2 + fwd_z ** 2)
    if fwd_len < 0.01:  # anchor and target too close
        return None

    fwd_x /= fwd_len
    fwd_z /= fwd_len

    # Right direction (perpendicular in XZ plane)
    right_x = fwd_z
    right_z = -fwd_x

    # Query relative to anchor
    rel_x = qx - ax
    rel_y = qy - ay
    rel_z = qz - az

    # Project onto person's coordinate frame
    forward_proj = rel_x * fwd_x + rel_z * fwd_z
    right_proj = rel_x * right_x + rel_z * right_z
    up_proj = rel_y  # Y is vertical

    dist_horiz = math.sqrt(forward_proj ** 2 + right_proj ** 2)
    if dist_horiz < 0.05:  # too close
        return None

    # Compute horizontal angle (0=front, clockwise)
    angle = math.degrees(math.atan2(right_proj, forward_proj))
    angle = angle % 360
    direction = angle_to_direction_8(angle)

    # Add vertical component if significant
    vertical = None
    if abs(up_proj) > 0.3 and abs(up_proj) > dist_horiz * 0.3:
        vertical = "above" if up_proj > 0 else "below"

    return {
        "direction": direction,
        "vertical": vertical,
        "forward": forward_proj,
        "right": right_proj,
        "up": up_proj,
    }


# ── Question templates ──

CAMERA_TEMPLATES = [
    "Where is {target} relative to {anchor} in the image?",
    "What is the position of {target} compared to {anchor}?",
    "Could you tell me the location of {target} in comparison to {anchor}?",
    "How is {target} positioned with respect to {anchor}?",
    "In this image, where is {target} in relation to {anchor}?",
    "Relative to {anchor}, where is {target}?",
    "Looking at the image, is {target} to the left, right, above, or below {anchor}?",
    "Describe the spatial relationship between {target} and {anchor}.",
]

SIMULATION_TEMPLATES = [
    "If you stand at {anchor} facing {facing}, where is {query}?",
    "Imagine standing at {anchor} looking towards {facing}, where is {query}?",
    "Standing at {anchor} and facing {facing}, in which direction is {query}?",
    "If you are at {anchor} looking at {facing}, where would {query} be?",
    "From {anchor}'s position facing {facing}, where is {query} located?",
]


def make_label(obj, all_objs_in_frame=None):
    """Create a readable label, disambiguating same-label objects if needed."""
    label = obj["label"]
    if all_objs_in_frame:
        same_label = [o for o in all_objs_in_frame if o["label"] == label]
        if len(same_label) > 1:
            # Find position of this object among same-label objects (by object_id)
            same_label_sorted = sorted(same_label, key=lambda o: o["object_id"])
            idx = next(i for i, o in enumerate(same_label_sorted)
                       if o["object_id"] == obj["object_id"])
            ordinals = ["first", "second", "third", "fourth", "fifth"]
            if idx < len(ordinals):
                return f"the {ordinals[idx]} {label}"
            return f"{label} #{idx + 1}"
    return f"the {label}"


def generate_choices(correct, n_choices=4):
    """Generate multiple choice options with the correct answer."""
    all_dirs = ["left", "right", "front", "back", "above", "below",
                "front-left", "front-right", "back-left", "back-right"]
    distractors = [d for d in all_dirs if d != correct]
    random.shuffle(distractors)
    choices = [correct] + distractors[: n_choices - 1]
    random.shuffle(choices)

    labels = ["A", "B", "C", "D"]
    choice_strs = [f"{labels[i]}. {choices[i]}" for i in range(len(choices))]
    correct_label = labels[choices.index(correct)]

    return {
        "choices": "\n".join(choice_strs),
        "answer": f"{correct_label}. {correct}",
    }


# ── Main generation ──

def load_scene_data(data_dir):
    """Load all visibility data, organized by scene."""
    scenes_dir = os.path.join(data_dir, "scannetv2_val")
    scenes = {}

    for scene_name in sorted(os.listdir(scenes_dir)):
        scene_path = os.path.join(scenes_dir, scene_name)
        vis_dir = os.path.join(scene_path, "visibility_data")
        img_dir = os.path.join(scene_path, "original_images")

        if not os.path.isdir(vis_dir) or not os.path.isdir(img_dir):
            continue

        frames = []
        import glob
        for vf in sorted(glob.glob(os.path.join(vis_dir, "*_visibility.json"))):
            with open(vf) as f:
                frame_data = json.load(f)

            # Get image filename
            img_name = frame_data["image_path"]
            img_path = os.path.join(
                "scannetv2_val", scene_name, "original_images", img_name
            )

            # Filter well-visible objects
            good_objs = [
                o
                for o in frame_data["visible_objects"]
                if o["visibility_ratio"] >= 0.25 and o["visible_points_count"] >= 2
            ]

            if len(good_objs) >= 2:
                frames.append(
                    {
                        "image_path": img_path,
                        "image_name": img_name,
                        "objects": good_objs,
                    }
                )

        if frames:
            scenes[scene_name] = frames

    return scenes


def load_benchmark_images(data_dir):
    """Load set of image paths used by benchmark, to exclude from training."""
    bench_path = os.path.join(data_dir, "ViewSpatial-Bench.json")
    if not os.path.exists(bench_path):
        return set()

    with open(bench_path) as f:
        data = json.load(f)

    bench_images = set()
    for d in data:
        for p in d["image_path"]:
            bench_images.add(p.replace("ViewSpatial-Bench/", ""))
    return bench_images


def generate_camera_qa(frame, obj_a, obj_b, idx):
    """Generate a camera perspective relative direction QA."""
    rel = compute_2d_relation(obj_a, obj_b)
    if rel is None:
        return None

    all_objs = frame["objects"]
    template = random.choice(CAMERA_TEMPLATES)
    question = template.format(
        target=make_label(obj_b, all_objs),
        anchor=make_label(obj_a, all_objs),
    )

    choice_data = generate_choices(rel["direction"])

    return {
        "id": f"gen_cam_{idx:06d}",
        "source": "viewspatial_generated",
        "images": [os.path.join("/workspace/datasets/viewspatial", frame["image_path"])],
        "num_views": 1,
        "question": question,
        "answer_type": "multi_choice",
        "choices": choice_data["choices"].split("\n"),
        "answer": choice_data["answer"],
        "frame_type": "camera",
        "pair_id": None,  # filled later
        "relation_label": None,
        "split": "train",
        "_scene": None,  # internal, removed later
        "_obj_pair": None,
    }


def generate_simulation_qa(frame, anchor_obj, facing_obj, query_obj, idx):
    """Generate a scene simulation (person perspective) QA."""
    rel = compute_3d_simulation(anchor_obj, facing_obj, query_obj)
    if rel is None:
        return None

    direction = rel["direction"]
    if rel["vertical"] and random.random() < 0.3:
        direction = rel["vertical"]

    all_objs = frame["objects"]
    template = random.choice(SIMULATION_TEMPLATES)
    question = template.format(
        anchor=make_label(anchor_obj, all_objs),
        facing=make_label(facing_obj, all_objs),
        query=make_label(query_obj, all_objs),
    )

    choice_data = generate_choices(direction)

    return {
        "id": f"gen_sim_{idx:06d}",
        "source": "viewspatial_generated",
        "images": [os.path.join("/workspace/datasets/viewspatial", frame["image_path"])],
        "num_views": 1,
        "question": question,
        "answer_type": "multi_choice",
        "choices": choice_data["choices"].split("\n"),
        "answer": choice_data["answer"],
        "frame_type": "person",
        "pair_id": None,
        "relation_label": None,
        "split": "train",
        "_scene": None,
        "_obj_pair": None,
    }


def generate_all(data_dir, output_dir, target_size=50000, seed=42):
    random.seed(seed)

    print("Loading scene data...")
    scenes = load_scene_data(data_dir)
    print(f"Loaded {len(scenes)} scenes")

    bench_images = load_benchmark_images(data_dir)
    print(f"Excluding {len(bench_images)} benchmark images")

    camera_samples = []
    simulation_samples = []
    cam_idx = 0
    sim_idx = 0

    # Target: ~60% camera, ~40% simulation
    cam_target = int(target_size * 0.6)
    sim_target = int(target_size * 0.4)

    for scene_name, frames in scenes.items():
        # Filter out benchmark frames
        train_frames = [
            f for f in frames if f["image_path"] not in bench_images
        ]

        if not train_frames:
            continue

        for frame in train_frames:
            objs = frame["objects"]
            if len(objs) < 2:
                continue

            # Camera perspective: for each pair of objects
            for obj_a, obj_b in combinations(objs, 2):
                # Skip same-label pairs (hard to disambiguate in text)
                if obj_a["label"] == obj_b["label"]:
                    continue

                qa = generate_camera_qa(frame, obj_a, obj_b, cam_idx)
                if qa:
                    qa["_scene"] = scene_name
                    qa["_obj_pair"] = (
                        min(obj_a["object_id"], obj_b["object_id"]),
                        max(obj_a["object_id"], obj_b["object_id"]),
                    )
                    camera_samples.append(qa)
                    cam_idx += 1

            # Scene simulation: for triplets of objects
            if len(objs) >= 3:
                # Sample triplets to control quantity
                obj_list = list(objs)
                max_triplets = min(6, len(obj_list) * (len(obj_list) - 1))
                sampled_triplets = []

                for _ in range(max_triplets):
                    triple = random.sample(obj_list, 3)
                    sampled_triplets.append(triple)

                for anchor, facing, query in sampled_triplets:
                    qa = generate_simulation_qa(
                        frame, anchor, facing, query, sim_idx
                    )
                    if qa:
                        qa["_scene"] = scene_name
                        qa["_obj_pair"] = (
                            anchor["object_id"],
                            query["object_id"],
                        )
                        simulation_samples.append(qa)
                        sim_idx += 1

    print(f"Generated {len(camera_samples)} camera QA, {len(simulation_samples)} simulation QA")

    # Subsample to target size
    if len(camera_samples) > cam_target:
        camera_samples = random.sample(camera_samples, cam_target)
    if len(simulation_samples) > sim_target:
        simulation_samples = random.sample(simulation_samples, sim_target)

    # ── Build consistency pairs ──
    # Match camera and simulation QA from same scene
    cam_by_scene = defaultdict(list)
    sim_by_scene = defaultdict(list)

    for s in camera_samples:
        cam_by_scene[s["_scene"]].append(s)
    for s in simulation_samples:
        sim_by_scene[s["_scene"]].append(s)

    pair_count = 0
    for scene_name in cam_by_scene:
        cam_list = cam_by_scene[scene_name]
        sim_list = sim_by_scene.get(scene_name, [])

        # Pair by matching object pairs or just by scene
        random.shuffle(cam_list)
        random.shuffle(sim_list)

        n_pairs = min(len(cam_list), len(sim_list))
        for i in range(n_pairs):
            cam_list[i]["pair_id"] = sim_list[i]["id"]
            sim_list[i]["pair_id"] = cam_list[i]["id"]
            pair_count += 1

    print(f"Created {pair_count} consistency pairs")

    # Combine and clean
    all_samples = camera_samples + simulation_samples
    random.shuffle(all_samples)

    # Remove internal fields
    for s in all_samples:
        s.pop("_scene", None)
        s.pop("_obj_pair", None)

    # Write output
    os.makedirs(output_dir, exist_ok=True)

    # Write main training file
    train_path = os.path.join(output_dir, "viewspatial_train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write consistency pairs
    pairs = []
    id_to_sample = {s["id"]: s for s in all_samples}
    seen = set()
    for s in all_samples:
        if s["pair_id"] and s["id"] not in seen:
            partner = id_to_sample.get(s["pair_id"])
            if partner:
                pairs.append(
                    {
                        "pair_id": f"pair_{len(pairs):06d}",
                        "sample_a": s,
                        "sample_b": partner,
                        "frame_a": s["frame_type"],
                        "frame_b": partner["frame_type"],
                    }
                )
                seen.add(s["id"])
                seen.add(partner["id"])

    pairs_path = os.path.join(output_dir, "consistency_pairs.jsonl")
    with open(pairs_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Stats
    frame_counts = defaultdict(int)
    for s in all_samples:
        frame_counts[s["frame_type"]] += 1

    print(f"\n=== Generation Summary ===")
    print(f"Total training samples: {len(all_samples)}")
    print(f"Frame types: {dict(frame_counts)}")
    print(f"Consistency pairs: {len(pairs)}")
    print(f"Output: {train_path}")
    print(f"Pairs: {pairs_path}")

    return all_samples, pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/workspace/datasets/viewspatial")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--target_size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_all(args.data_dir, args.output_dir, args.target_size, args.seed)
