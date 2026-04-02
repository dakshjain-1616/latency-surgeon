# LatencySurgeon Usage Guide

## Installation

```bash
pip install -r requirements.txt
```

## CLI Commands

### Decompose a Model

```bash
python -m latency_surgeon decompose --model bert-base-uncased --rank 64 --output ./optimized
```

### Analyze Model Structure

```bash
python -m latency_surgeon analyze --model bert-base-uncased
```

### Benchmark Performance

```bash
python -m latency_surgeon benchmark --model bert-base-uncased --decomposed
```

### Validate Decomposition

```bash
python -m latency_surgeon validate --tensor-shape 10,10,10 --rank 5
```

### Compare Models

```bash
python -m latency_surgeon compare --model bert-base-uncased --rank 64
```

## Library Usage

```python
from latency_surgeon import TuckerDecomposition, optimize_model
from transformers import AutoModel

# Tucker decomposition
tucker = TuckerDecomposition(rank=(64, 64, 64))
core, factors = tucker.decompose(tensor)

# Model optimization
model = AutoModel.from_pretrained('bert-base-uncased')
optimized = optimize_model(model, rank=64)
```

## Expected Performance

- **BERT-base**: ~40% latency reduction
- **Compression**: 60-65% parameter reduction
- **Accuracy**: Minimal loss (<2% on most tasks)