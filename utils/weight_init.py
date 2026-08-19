"""Reusable weight initialization functions for MV-SRBF models.

Provides mainstream initialization strategies used in modern CNN and Transformer
architectures. Follows conventions from:

- He et al. (2015): Kaiming/He initialization for ReLU networks (ResNet).
- Glorot & Bengio (2010): Xavier/Glorot for sigmoid/tanh networks.
- Dosovitskiy et al. (2020): Truncated normal (std=0.02) for ViT.
- Devlin et al. (2018): Truncated normal (std=0.02) for BERT.
- timm library: Industry-standard truncated normal for Vision Transformers.
- PyTorch defaults: Kaiming uniform with a=sqrt(5), mode=fan_in.
"""
from __future__ import annotations

import torch.nn as nn


# ---------------------------------------------------------------------------
# Core initialization functions
# ---------------------------------------------------------------------------


def init_trunc_normal(module: nn.Module, std: float = 0.02, bias_zero: bool = True) -> None:
    """Truncated-normal initialization — the standard for Vision Transformers.

    Used by ViT, DeiT, BERT, and most HuggingFace transformer models.
    Truncates to [-2*std, 2*std] to avoid extreme outlier weights.

    Applied to:
        - Linear: weight ~ TruncatedNormal(0, std), bias = 0
        - Conv2d / Conv1d: weight ~ TruncatedNormal(0, std), bias = 0
        - LayerNorm / GroupNorm / BatchNorm: weight = 1, bias = 0
        - Embedding: weight ~ Normal(0, std)

    Args:
        module: The module to initialize.
        std: Standard deviation (default 0.02 per ViT/BERT convention).
        bias_zero: If True, zero-initialize biases.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.trunc_normal_(module.weight, std=std, a=-2.0 * std, b=2.0 * std)
        if bias_zero and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=std)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def init_kaiming(
    module: nn.Module,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    distribution: str = "normal",
    bias_zero: bool = True,
) -> None:
    """Kaiming (He) initialization for ReLU-activated networks.

    Standard for ResNet, EfficientNet, and most CNN architectures.

    Applied to:
        - Linear: Kaiming init with configurable mode/nonlinearity.
        - Conv: Kaiming init with configurable mode/nonlinearity.
        - BatchNorm / GroupNorm / LayerNorm: weight = 1, bias = 0.

    Args:
        module: The module to initialize.
        mode: ``"fan_in"`` (default, standard) or ``"fan_out"`` (ResNet shortcut).
            - ``fan_in``: preserves variance of forward pass (default for most layers).
            - ``fan_out``: preserves variance of backward pass (used by ResNet for Conv).
        nonlinearity: ``"relu"`` (default), ``"leaky_relu"``, ``"linear"``, etc.
            Controls the ``a`` parameter in Kaiming init.
        distribution: ``"normal"`` (default) or ``"uniform"``.
        bias_zero: If True, zero-initialize biases.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        weight = module.weight
        if weight is not None:
            if distribution == "normal":
                nn.init.kaiming_normal_(weight, a=_relu_slope(nonlinearity), mode=mode, nonlinearity=nonlinearity)
            else:
                nn.init.kaiming_uniform_(weight, a=_relu_slope(nonlinearity), mode=mode, nonlinearity=nonlinearity)
        if bias_zero and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def init_xavier(
    module: nn.Module,
    gain: float = 1.0,
    distribution: str = "normal",
    bias_zero: bool = True,
) -> None:
    """Xavier (Glorot) initialization for sigmoid/tanh-activated networks.

    Suitable for attention mechanisms and pre-activation residual networks
    that use non-ReLU activations.

    Applied to:
        - Linear: Xavier init.
        - Conv: Xavier init.
        - BatchNorm / GroupNorm / LayerNorm: weight = 1, bias = 0.

    Args:
        module: The module to initialize.
        gain: Scaling factor (default 1.0). Use sqrt(2) for ReLU, 1.0 for sigmoid/tanh.
        distribution: ``"normal"`` or ``"uniform"``.
        bias_zero: If True, zero-initialize biases.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        weight = module.weight
        if weight is not None:
            if distribution == "normal":
                nn.init.xavier_normal_(weight, gain=gain)
            else:
                nn.init.xavier_uniform_(weight, gain=gain)
        if bias_zero and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def init_classifier(module: nn.Module, std: float = 0.001) -> None:
    """Small-normal initialization for classification heads.

    Uses a small std to prevent large initial logits that could destabilize
    early training. Common for ReID identity classifiers.

    Args:
        module: The module to initialize.
        std: Standard deviation (default 0.001).
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Model-level initialization
# ---------------------------------------------------------------------------


def apply_model_init(model: nn.Module, strategy: str = "kaiming", **kwargs) -> nn.Module:
    """Apply weight initialization to the entire model.

    Args:
        model: The PyTorch model to initialize.
        strategy: Initialization strategy. Options:
            - ``"kaiming"``: He/Kaiming init (default, best for CNN backbones with ReLU).
              Extra kwargs: ``mode``, ``nonlinearity``, ``distribution``.
            - ``"trunc_normal"``: Truncated normal (best for ViT/transformer backbones).
              Extra kwargs: ``std`` (default 0.02).
            - ``"xavier"``: Xavier/Glorot init (for sigmoid/tanh networks).
              Extra kwargs: ``gain``, ``distribution``.
            - ``"classifier"``: Small-normal for classification heads.
              Extra kwargs: ``std``.

    Returns:
        The initialized model (for chaining).

    Examples::

        # ResNet backbone (CNN, ReLU)
        apply_model_init(resnet, "kaiming", mode="fan_out", nonlinearity="relu")

        # ViT backbone (Transformer)
        apply_model_init(vit, "trunc_normal", std=0.02)

        # Classification head
        apply_model_init(classifier, "classifier", std=0.001)
    """
    if strategy == "kaiming":
        model.apply(_make_kaiming_init(**kwargs))
    elif strategy == "trunc_normal":
        model.apply(_make_trunc_normal_init(**kwargs))
    elif strategy == "xavier":
        model.apply(_make_xavier_init(**kwargs))
    elif strategy == "classifier":
        model.apply(_make_classifier_init(**kwargs))
    else:
        raise ValueError(
            f"Unknown init strategy: {strategy!r}. "
            "Use 'kaiming', 'trunc_normal', 'xavier', or 'classifier'."
        )
    return model


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _relu_slope(nonlinearity: str) -> float:
    """Return the negative slope for common nonlinearities."""
    slopes = {
        "relu": 0.0,
        "leaky_relu": 0.01,
        "prelu": 0.25,
        "rrelu": 1.0 / 3.0,
        "linear": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
    }
    return slopes.get(nonlinearity, 0.0)


def _make_kaiming_init(**kwargs):
    mode = kwargs.get("mode", "fan_in")
    nonlinearity = kwargs.get("nonlinearity", "relu")
    distribution = kwargs.get("distribution", "normal")

    def _init(module: nn.Module) -> None:
        init_kaiming(module, mode=mode, nonlinearity=nonlinearity, distribution=distribution)

    return _init


def _make_trunc_normal_init(**kwargs):
    std = kwargs.get("std", 0.02)

    def _init(module: nn.Module) -> None:
        init_trunc_normal(module, std=std)

    return _init


def _make_xavier_init(**kwargs):
    gain = kwargs.get("gain", 1.0)
    distribution = kwargs.get("distribution", "normal")

    def _init(module: nn.Module) -> None:
        init_xavier(module, gain=gain, distribution=distribution)

    return _init


def _make_classifier_init(**kwargs):
    std = kwargs.get("std", 0.001)

    def _init(module: nn.Module) -> None:
        init_classifier(module, std=std)

    return _init


