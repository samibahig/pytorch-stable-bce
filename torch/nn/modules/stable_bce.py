import torch
import torch.nn as nn
import torch.nn.functional as F


class StableWeightedBCEWithLogits(nn.Module):
    """Stable Weighted Binary Cross Entropy Loss with optional label smoothing.

    This loss combines a Sigmoid layer and the Binary Cross Entropy loss in one
    single class, using the numerically stable log-sum-exp trick internally
    (via :func:`torch.nn.functional.binary_cross_entropy_with_logits`).

    Extends :class:`~torch.nn.BCEWithLogitsLoss` with:

    * **pos_weight**: positive class weight to handle class imbalance.
    * **label_smoothing**: prevents overconfidence by mixing target labels
      toward 0.5, improving generalization on noisy or ambiguous datasets.

    This loss is suitable for:

    * Imbalanced binary classification (e.g., fraud detection, rare event modeling)
    * Multi-label classification
    * Medical imaging and anomaly detection

    Args:
        pos_weight (Tensor, optional): a weight of positive examples to be
            broadcast with target. Must be a tensor with length equal to the
            number of classes. Default: ``None``.
        label_smoothing (float, optional): a value in [0.0, 1.0] that
            specifies the amount of label smoothing when computing the loss.
            A value of 0.0 means no smoothing (default). When > 0, target
            values are shifted toward 0.5:
            ``y_smooth = y * (1 - label_smoothing) + 0.5 * label_smoothing``.
            Default: ``0.0``.

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar by default.

    Examples::

        >>> loss_fn = StableWeightedBCEWithLogits(label_smoothing=0.1)
        >>> input = torch.randn(8, requires_grad=True)
        >>> target = torch.randint(0, 2, (8,)).float()
        >>> loss = loss_fn(input, target)
        >>> loss.backward()

        >>> # Imbalanced dataset: weight positives 3x
        >>> pos_weight = torch.tensor([3.0])
        >>> loss_fn = StableWeightedBCEWithLogits(pos_weight=pos_weight)
        >>> loss = loss_fn(input, target)
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= label_smoothing <= 1.0):
            raise ValueError(
                f"label_smoothing must be in [0.0, 1.0], got {label_smoothing}"
            )
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the stable weighted BCE loss.

        Args:
            inputs (Tensor): Raw (unnormalized) logits of shape :math:`(N, *)`.
            targets (Tensor): Binary labels of shape :math:`(N, *)`,
                with values in {0, 1}.

        Returns:
            Tensor: Scalar loss value.
        """
        targets = targets.float()

        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        return F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.pos_weight,
            reduction="mean",
        )
