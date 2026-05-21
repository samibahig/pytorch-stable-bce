"""Tests for StableWeightedBCEWithLogits — 32 test cases."""

import pytest
import torch
import torch.nn as nn

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from torch.nn.modules.stable_bce import StableWeightedBCEWithLogits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(n=8, *, requires_grad=True):
    x = torch.randn(n, requires_grad=requires_grad)
    y = torch.randint(0, 2, (n,)).float()
    return x, y


# ---------------------------------------------------------------------------
# 1. Construction & repr
# ---------------------------------------------------------------------------

def test_construct_default():
    loss_fn = StableWeightedBCEWithLogits()
    assert loss_fn.label_smoothing == 0.0
    assert loss_fn.pos_weight is None
    assert loss_fn.reduction == "mean"


def test_construct_label_smoothing():
    loss_fn = StableWeightedBCEWithLogits(label_smoothing=0.1)
    assert loss_fn.label_smoothing == 0.1


def test_construct_pos_weight():
    pw = torch.tensor([3.0])
    loss_fn = StableWeightedBCEWithLogits(pos_weight=pw)
    assert loss_fn.pos_weight is not None


def test_extra_repr_no_smoothing():
    loss_fn = StableWeightedBCEWithLogits()
    assert "label_smoothing=0.0" in repr(loss_fn)


def test_extra_repr_with_smoothing_and_weight():
    pw = torch.tensor([2.0])
    loss_fn = StableWeightedBCEWithLogits(pos_weight=pw, label_smoothing=0.05)
    r = repr(loss_fn)
    assert "label_smoothing=0.05" in r
    assert "pos_weight" in r


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------

def test_invalid_label_smoothing_above_one():
    with pytest.raises(ValueError):
        StableWeightedBCEWithLogits(label_smoothing=1.0)


def test_invalid_label_smoothing_negative():
    with pytest.raises(ValueError):
        StableWeightedBCEWithLogits(label_smoothing=-0.1)


def test_invalid_label_smoothing_way_above():
    with pytest.raises(ValueError):
        StableWeightedBCEWithLogits(label_smoothing=2.5)


# ---------------------------------------------------------------------------
# 3. Forward — output shape and type
# ---------------------------------------------------------------------------

def test_forward_scalar_output():
    loss_fn = StableWeightedBCEWithLogits()
    x, y = make_inputs()
    assert loss_fn(x, y).dim() == 0


def test_forward_positive_loss():
    loss_fn = StableWeightedBCEWithLogits()
    x, y = make_inputs()
    assert loss_fn(x, y).item() > 0


def test_forward_not_nan():
    loss_fn = StableWeightedBCEWithLogits()
    x, y = make_inputs()
    assert not torch.isnan(loss_fn(x, y))


def test_forward_not_inf():
    loss_fn = StableWeightedBCEWithLogits()
    x, y = make_inputs()
    assert not torch.isinf(loss_fn(x, y))


def test_forward_multilabel_2d():
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.randn(4, 10, requires_grad=True)
    y = torch.randint(0, 2, (4, 10)).float()
    loss = loss_fn(x, y)
    assert loss.dim() == 0


def test_forward_single_element():
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.randn(1, requires_grad=True)
    y = torch.tensor([1.0])
    assert loss_fn(x, y).dim() == 0


# ---------------------------------------------------------------------------
# 4. Numerical stability
# ---------------------------------------------------------------------------

def test_extreme_positive_logits():
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.tensor([1e6, 1e6], requires_grad=True)
    y = torch.tensor([1.0, 0.0])
    loss = loss_fn(x, y)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_extreme_negative_logits():
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.tensor([-1e6, -1e6], requires_grad=True)
    y = torch.tensor([1.0, 0.0])
    loss = loss_fn(x, y)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_all_zeros_target():
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.randn(8, requires_grad=True)
    y = torch.zeros(8)
    loss = loss_fn(x, y)
    assert not torch.isnan(loss)


def test_all_ones_target():
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.randn(8, requires_grad=True)
    y = torch.ones(8)
    loss = loss_fn(x, y)
    assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# 5. Equivalence with BCEWithLogitsLoss
# ---------------------------------------------------------------------------

def test_equivalence_no_smoothing():
    """With label_smoothing=0.0 must match BCEWithLogitsLoss exactly."""
    x = torch.randn(32)
    y = torch.randint(0, 2, (32,)).float()
    ref = nn.BCEWithLogitsLoss()(x, y)
    ours = StableWeightedBCEWithLogits(label_smoothing=0.0)(x, y)
    torch.testing.assert_close(ours, ref)


def test_equivalence_pos_weight():
    """With pos_weight and no smoothing must match BCEWithLogitsLoss."""
    x = torch.randn(16)
    y = torch.randint(0, 2, (16,)).float()
    pw = torch.tensor([2.0])
    ref = nn.BCEWithLogitsLoss(pos_weight=pw)(x, y)
    ours = StableWeightedBCEWithLogits(pos_weight=pw)(x, y)
    torch.testing.assert_close(ours, ref)


# ---------------------------------------------------------------------------
# 6. Label smoothing behaviour
# ---------------------------------------------------------------------------

def test_label_smoothing_changes_loss():
    x = torch.randn(16)
    y = torch.randint(0, 2, (16,)).float()
    loss_no = StableWeightedBCEWithLogits(label_smoothing=0.0)(x, y)
    loss_sm = StableWeightedBCEWithLogits(label_smoothing=0.1)(x, y)
    assert not torch.isclose(loss_no, loss_sm), "Smoothing should change the loss"


def test_label_smoothing_valid_range():
    loss_fn = StableWeightedBCEWithLogits(label_smoothing=0.2)
    x, y = make_inputs()
    loss = loss_fn(x, y)
    assert not torch.isnan(loss) and not torch.isinf(loss)


# ---------------------------------------------------------------------------
# 7. Backward pass
# ---------------------------------------------------------------------------

def test_backward_basic():
    loss_fn = StableWeightedBCEWithLogits()
    x, y = make_inputs()
    loss_fn(x, y).backward()
    assert x.grad is not None


def test_backward_no_nan_grad():
    loss_fn = StableWeightedBCEWithLogits()
    x, y = make_inputs()
    loss_fn(x, y).backward()
    assert not torch.isnan(x.grad).any()


def test_backward_no_inf_grad():
    loss_fn = StableWeightedBCEWithLogits()
    x, y = make_inputs()
    loss_fn(x, y).backward()
    assert not torch.isinf(x.grad).any()


def test_backward_extreme_logits():
    loss_fn = StableWeightedBCEWithLogits()
    x = torch.tensor([1e6, -1e6, 1e6, -1e6], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([1.0, 0.0, 0.0, 1.0])
    loss_fn(x, y).backward()
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


def test_backward_label_smoothing():
    loss_fn = StableWeightedBCEWithLogits(label_smoothing=0.2)
    x, y = make_inputs()
    loss_fn(x, y).backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# 8. Buffer registration (pos_weight moves with .to())
# ---------------------------------------------------------------------------

def test_pos_weight_is_buffer():
    pw = torch.tensor([2.0])
    loss_fn = StableWeightedBCEWithLogits(pos_weight=pw)
    buffers = dict(loss_fn.named_buffers())
    assert "pos_weight" in buffers


def test_state_dict_contains_pos_weight():
    pw = torch.tensor([3.0])
    loss_fn = StableWeightedBCEWithLogits(pos_weight=pw)
    sd = loss_fn.state_dict()
    assert "pos_weight" in sd


def test_no_pos_weight_absent_from_state_dict():
    loss_fn = StableWeightedBCEWithLogits()
    sd = loss_fn.state_dict()
    assert "pos_weight" not in sd or sd.get("pos_weight") is None
