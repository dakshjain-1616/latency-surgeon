# HuggingFace Export Guide

Export your Tucker-compressed model to HuggingFace Hub in one command.

## Quick Export

```bash
# 1. Apply surgery and save locally
latency-surgeon surgery gpt2 --rank 64 --output ./compressed_model

# 2. Push to Hub
python hf_export/push_to_hub.py \
    --model-path ./compressed_model \
    --repo-id your-username/gpt2-tucker-r64 \
    --original-model gpt2 \
    --rank 64 \
    --speedup 1.4 \
    --perplexity-delta 0.03

# Or set token via env variable
HF_TOKEN=hf_xxxx python hf_export/push_to_hub.py ...
```

## Python API

```python
from hf_export.push_to_hub import push_to_hub

url = push_to_hub(
    model_path="./compressed_model",
    repo_id="your-username/gpt2-tucker-r64",
    original_model="gpt2",
    rank=64,
    speedup=1.4,
    perplexity_delta=0.03,
)
print(f"Model at: {url}")
```

## What Gets Uploaded

| File | Description |
|:-----|:------------|
| `pytorch_model.bin` | Tucker-compressed model weights |
| `config.json` | Model architecture config |
| `tokenizer.json` | Original tokenizer |
| `surgery_config.json` | Compression metadata (rank, speedup, etc.) |
| `README.md` | Auto-generated model card |

## Loading a Pushed Model

```python
from latency_surgeon.hf_integration import load_compressed_model

model, tokenizer = load_compressed_model(
    "your-username/gpt2-tucker-r64",
    rank=64,
)
```

## config.json Reference

See `config.json` in this directory for the full schema of compression metadata
stored alongside the model.
