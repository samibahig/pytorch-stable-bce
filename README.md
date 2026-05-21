# StableWeightedBCEWithLogits — PyTorch Contribution

A numerically stable binary cross-entropy loss with optional **label smoothing** and **class imbalance weighting**, ready to submit as a pull request to [pytorch/pytorch](https://github.com/pytorch/pytorch).

---

## What it adds

| Feature | Description |
|---|---|
| Numerical stability | Uses `F.binary_cross_entropy_with_logits` (log-sum-exp trick) |
| `pos_weight` | Upweight positive class for imbalanced datasets |
| `label_smoothing` | Smooths targets toward 0.5 for robustness |
| PyTorch-style API | Drop-in alongside `BCEWithLogitsLoss` |

---

## Usage

```python
from torch.nn.modules.stable_bce import StableWeightedBCEWithLogits
import torch

# Basic
loss_fn = StableWeightedBCEWithLogits()

# With label smoothing
loss_fn = StableWeightedBCEWithLogits(label_smoothing=0.1)

# With class imbalance weighting (positives weighted 3x)
pos_weight = torch.tensor([3.0])
loss_fn = StableWeightedBCEWithLogits(pos_weight=pos_weight)

# Forward + backward
x = torch.randn(8, requires_grad=True)
y = torch.randint(0, 2, (8,)).float()
loss = loss_fn(x, y)
loss.backward()
```

---

## Files to add to pytorch/pytorch

| File in this repo | Target in pytorch/pytorch |
|---|---|
| `torch/nn/modules/stable_bce.py` | Add class into `torch/nn/modules/loss.py` |
| `test/test_stable_bce.py` | `test/test_nn.py` or new `test/test_stable_bce.py` |

---

## Run tests locally

```bash
pip install pytest torch
pytest test/test_stable_bce.py -v
```


## PR template

```markdown
## Summary
Adds `StableWeightedBCEWithLogits`, a stable weighted binary cross-entropy loss
with optional label smoothing.

## Motivation
`BCEWithLogitsLoss` does not support label smoothing. This PR adds a minimal,
stable extension for:
- class-imbalanced binary classification
- multi-label classification
- medical imaging and anomaly detection

## Changes
- Adds `StableWeightedBCEWithLogits` module in `torch/nn/modules/loss.py`
- Supports `pos_weight` for imbalance handling
- Adds `label_smoothing` parameter with input validation
- Adds tests: forward, backward, extreme logits, equivalence with BCEWithLogitsLoss

## Testing
All tests pass: `pytest test/test_stable_bce.py -v`

## Use cases
- Medical classification
- Anomaly detection
- Multi-label learning
```
