import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class StableWeightedBCEWithLogits(nn.modules.loss._WeightedLoss):
    r"""Numerically stable binary cross-entropy loss with optional label smoothing
    and positive-class weighting.

    Drop-in companion to :class:`~torch.nn.BCEWithLogitsLoss`, designed for:

    * Class-imbalanced binary classification (e.g., fraud, anomaly detection)
    * Multi-label classification
    * Medical imaging

    The loss is computed as:

    .. math::

        \ell(x, y) = -\frac{1}{N} \sum_{i=1}^{N}
            \Bigl[
                w \cdot \tilde{y}_i \log \sigma(x_i)
              + (1 - \tilde{y}_i) \log (1 - \sigma(x_i))
            \Bigr]

    where :math:`\tilde{y}_i = y_i (1 - \varepsilon) + 0.5\,\varepsilon` when
    ``label_smoothing`` :math:`\varepsilon > 0`, otherwise :math:`\tilde{y}_i = y_i`,
    and :math:`w` is ``pos_weight``.

    Internally the log-sum-exp trick from
    :func:`~torch.nn.functional.binary_cross_entropy_with_logits` is used, so the
    computation is numerically safe even at large logit magnitudes.

    Args:
        pos_weight (Tensor, optional): Scalar or per-class weight for the positive
            class. Registered as a buffer so it moves with the module on
            ``.to(device)`` / ``.cuda()`` calls. Default: ``None``.
        label_smoothing (float): Amount of label smoothing in :math:`[0, 1)`.
            ``0.0`` (default) means no smoothing.  Acts as the binary analogue of
            ``CrossEntropyLoss(label_smoothing=…)``.

    Shape:
        - **Input** :math:`(*)` — raw (un-normalised) logits, any shape.
        - **Target** :math:`(*)` — binary labels :math:`\in \{0, 1\}`, same shape.
        - **Output** — scalar tensor.

    Examples::

        >>> import torch
        >>> from torch.nn import StableWeightedBCEWithLogits

        >>> # Basic usage
        >>> loss_fn = StableWeightedBCEWithLogits()
        >>> x = torch.randn(8, requires_grad=True)
        >>> y = torch.randint(0, 2, (8,)).float()
        >>> loss = loss_fn(x, y)
        >>> loss.backward()

        >>> # Label smoothing for noisy / ambiguous labels
        >>> loss_fn = StableWeightedBCEWithLogits(label_smoothing=0.1)

        >>> # 1:10 imbalanced dataset — weight positives 10x
        >>> pos_weight = torch.tensor([10.0])
        >>> loss_fn = StableWeightedBCEWithLogits(pos_weight=pos_weight)
        >>> loss_fn.pos_weight.device  # moves with the module
        device(type='cpu')
    """

    __constants__ = ["label_smoothing"]

    def __init__(
        self,
        pos_weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(weight=None, size_average=None, reduce=None, reduction="mean")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0.0, 1.0), got {label_smoothing}"
            )
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", pos_weight)

    def extra_repr(self) -> str:
        parts = [f"label_smoothing={self.label_smoothing}"]
        if self.pos_weight is not None:
            parts.append(f"pos_weight={self.pos_weight.tolist()}")
        return ", ".join(parts)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute the stable weighted BCE loss.

        Args:
            inputs (Tensor): Raw logits of shape :math:`(N, *)`.
            targets (Tensor): Binary labels of shape :math:`(N, *)`,
                values in :math:`\\{0, 1\\}`.

        Returns:
            Tensor: Scalar loss.
        """
        targets = targets.float()

        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        return F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
