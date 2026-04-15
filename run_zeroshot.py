import json, torch, os
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from collections import defaultdict

print('加载模型...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    '/workspace/models/qwen25-vl-7b',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    attn_implementation='eager'
)
processor = AutoProcessor.from_pretrained('/workspace/models/qwen25-vl-7b')
model.eval()

ds = load_dataset('lidingm/ViewSpatial-Bench')['test']
base = '/workspace/datasets/viewspatial'

correct, total = 0, 0
results = []

for i, sample in enumerate(tqdm(ds, desc='Zero-shot eval')):
    img_paths = [os.path.join(base, p.replace('ViewSpatial-Bench/', '')) for p in sample['image_path']]

    content = [{'type': 'image', 'image': f'file://{p}'} for p in img_paths]
    content.append({'type': 'text', 'text': f"{sample['question']}\n{sample['choices']}\nAnswer with the correct option letter and content."})
    messages = [{'role': 'user', 'content': content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors='pt').to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32)
    pred = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    gt = sample['answer']
    gt_letter = gt.split('.')[0].strip()
    is_correct = gt_letter.lower() in pred.lower()[:5]

    correct += int(is_correct)
    total += 1
    results.append({'id': i, 'pred': pred, 'gt': gt, 'correct': is_correct, 'question_type': sample['question_type']})

    if i < 3:
        print(f'  Pred: {pred} | GT: {gt} | {"OK" if is_correct else "WRONG"}')

    if (i+1) % 500 == 0:
        print(f'  Progress: {correct}/{total} = {correct/total*100:.1f}%')

acc = correct / total * 100
print(f'\nZero-Shot Accuracy: {acc:.2f}% ({correct}/{total})')

stats = defaultdict(lambda: {'c': 0, 't': 0})
for r in results:
    ft = 'camera' if 'Camera' in r['question_type'] else 'person'
    stats[ft]['t'] += 1
    stats[ft]['c'] += int(r['correct'])
    stats[r['question_type']]['t'] += 1
    stats[r['question_type']]['c'] += int(r['correct'])

print('\n按 frame type:')
for k in ['camera', 'person']:
    s = stats[k]
    print(f'  {k}: {s["c"]/s["t"]*100:.2f}% ({s["c"]}/{s["t"]})')

print('\n按 question_type:')
for k, s in sorted(stats.items()):
    if k not in ['camera', 'person']:
        print(f'  {k}: {s["c"]/s["t"]*100:.2f}% ({s["c"]}/{s["t"]})')

os.makedirs('results/zeroshot', exist_ok=True)
with open('results/zeroshot/viewspatial.json', 'w') as f:
    json.dump({'accuracy': acc, 'results': results}, f)
print('\n结果保存到 results/zeroshot/viewspatial.json')
