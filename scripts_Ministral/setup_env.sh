#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx qwen38; then
  echo "Environment qwen38 already exists."
else
  conda env create -f "$HERE/environment.yml"
fi

eval "$(conda shell.bash hook)"
conda activate qwen38
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-cache-dir -r "$HERE/requirements.txt"
python -m pip check
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA validation failed.")
print("GPU:", torch.cuda.get_device_name(0))
PY
