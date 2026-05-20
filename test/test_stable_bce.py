"""Tests for StableWeightedBCEWithLogits."""

import pytest
import torch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from torch.nn.modules.stable_bce import StableWeightedBCEWithLogits


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_basic():
    """Loss is a scalar and positive for random inputs."""
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.randn(8, requires_grad=True)
    y = torch.randint(0, 2, (8,)).float()

    loss = loss_fn(x, y)

    assert loss.dim() == 0, "Loss must be a scalar"
    assert loss.item() > 0, "BCE loss must be positive"


def test_forward_label_smoothing():
    """Label smoothing should not raise and should return a valid scalar."""
    loss_fn = StableWeightedBCEWithLogits(label_smoothing=0.1)
    x = torch.randn(16, requires_grad=True)
    y = torch.randint(0, 2, (16,)).float()

    loss = loss_fn(x, y)

    assert loss.dim() == 0
    assert not torch.isnan(loss), "Loss must not be NaN with label smoothing"
    assert not torch.isinf(loss), "Loss must not be Inf with label smoothing"


def test_forward_pos_weight():
    """pos_weight should run without error and return a valid scalar."""
    pos_weight = torch.tensor([2.0])
    loss_fn = StableWeightedBCEWithLogits(pos_weight=pos_weight)
    x = torch.randn(8, requires_grad=True)
    y = torch.randint(0, 2, (8,)).float()

    loss = loss_fn(x, y)

    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_forward_multilabel():
    """Works for multi-label inputs of shape (N, C)."""
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.randn(4, 10, requires_grad=True)
    y = torch.randint(0, 2, (4, 10)).float()

    loss = loss_fn(x, y)

    assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def test_backward_basic():
    """Gradients are computed and finite."""
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.randn(8, requires_grad=True)
    y = torch.randint(0, 2, (8,)).float()

    loss = loss_fn(x, y)
    loss.backward()

    assert x.grad is not None, "Gradient must be computed"
    assert not torch.isnan(x.grad).any(), "Gradient must not contain NaN"
    assert not torch.isinf(x.grad).any(), "Gradient must not contain Inf"


def test_backward_label_smoothing():
    """Gradients are finite with label smoothing enabled."""
    loss_fn = StableWeightedBCEWithLogits(label_smoothing=0.2)
    x = torch.randn(8, requires_grad=True)
    y = torch.randint(0, 2, (8,)).float()

    loss = loss_fn(x, y)
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_backward_extreme_logits():
    """Gradients remain finite for very large logit values (numerical stability)."""
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.tensor([1e6, -1e6, 1e6, -1e6], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([1.0, 0.0, 0.0, 1.0])

    loss = loss_fn(x, y)
    loss.backward()

    assert not torch.isnan(x.grad).any(), "Gradient must be finite for extreme logits"
    assert not torch.isinf(x.grad).any()


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------

def test_no_smoothing_zero():
    """label_smoothing=0.0 is equivalent to standard BCEWithLogitsLoss."""
    x = torch.randn(32)
    y = torch.randint(0, 2, (32,)).float()

    loss_stable = StableWeightedBCEWithLogits(label_smoothing=0.0)(x, y)
    loss_reference = torch.nn.BCEWithLogitsLoss()(x, y)

    torch.testing.assert_close(loss_stable, loss_reference)


def test_invalid_label_smoothing_raises():
    """label_smoothing outside [0, 1] should raise ValueError."""
    with pytest.raises(ValueError):
        StableWeightedBCEWithLogits(label_smoothing=1.5)

    with pytest.raises(ValueError):
        StableWeightedBCEWithLogits(label_smoothing=-0.1)
