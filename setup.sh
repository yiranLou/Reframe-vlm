#!/bin/bash
set -e

echo "=== Setting up ReFrame-VLM environment ==="

# Create conda environment
conda create -n reframe python=3.10 -y
conda activate reframe

# Core dependencies
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.0
pip install peft==0.13.0
pip install accelerate==0.34.0
pip install deepspeed==0.15.0
pip install qwen-vl-utils
pip install datasets wandb pillow tqdm pyyaml

# Flash attention
pip install flash-attn --no-build-isolation

# Verify
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('Transformers OK')"
python -c "from peft import LoraConfig; print('PEFT OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

echo "=== Setup complete ==="
