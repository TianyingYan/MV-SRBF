# Configuration

This document describes the supported vehicle-oriented configuration surface.

## Dataset

```yaml
dataset:
  name: "MSVR310"
  root: "../datasets"
  num_modalities: 3
  modalities: {modal1: "RGB", modal2: "NIR", modal3: "TIR"}
```

MV-SRBF currently supports all four multispectral vehicle ReID datasets: `MSVR310`, `RGBNT100`, `RGBN300`, and `WMVEID863`.
The root is the parent directory containing those dataset folders.

Relative filesystem paths are resolved from the working directory captured
when `get_config` loads the configuration. They are not resolved from
`data/dataloaders.py`, the YAML file directory, or the wheel installation
directory. The shipped defaults assume launch from the MV-SRBF repository
root; absolute paths are used unchanged.

## Input

```yaml
input:
  size: "auto"
  padding: [5, 10]
  padding_fill: 0
  resize_interpolation: "auto"
  normalize:
    default: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  train_augmentation:
    shared:
      random_horizontal_flip: 0.5
      random_crop: true
    batch_augment:
      enable: true
      random_erasing:
        enable: true
        mode: "synchronous"
        p: 0.5
        scale: [0.02, 0.4]
        ratio: [0.3, 3.33]
```

`input.size: "auto"` means `[128, 256]`.

`resize_interpolation: "auto"` uses bicubic for ViT-like backbones and bilinear otherwise.

Random erasing modes:

- `synchronous`: the same erase rectangle is applied to every modality of one sample.
- `asynchronous`: each modality is erased independently.

## Backbone

```yaml
model:
  encoder:
    backbone:
      source: "openai-clip"
      name: "ViT-B/16"
      pretrained: true
      pretrained_weights_path: "../weights/clip_vit_b_16.pt"
      vit_stem:
        enabled: true
        mode: "auto"
        stride_ratio: 0.75
```

`source` can be `torchvision`, `timm`, `torch_hub`, `dinov2`, `dinov3`, `openai-clip`, `open-clip`, `huggingface-hub`, `transformers`, or `custom`.

DINOv2 uses an official Hub name such as `dinov2_vitb14`. DINOv3 weights are gated; use the official repository checkout and downloaded weight file:

```yaml
backbone:
  source: "dinov3"
  name: "dinov3_vitb16"
  repo: "../dinov3"
  repo_source: "local"
  pretrained: true
  pretrained_weights_path: "../weights/dinov3_vitb16_pretrain_lvd1689m.pth"
  memory:
    enabled: false
    aggregation: "mean"
```

Both versions consume their official `forward_features` outputs. DINOv3 Memory mode averages normalized CLS with register tokens, exposed by the official model as `x_storage_tokens`, when `memory.enabled: true`; disabled mode uses CLS only. Feature width does not change. This switch controls register-token readout and does not remove those tokens from the backbone. DINOv2 does not accept Memory mode. To complete the patch grid, required pixels are split symmetrically across top/bottom and left/right like YAML `input.padding`; an odd remainder differs by at most one pixel between opposite sides.

For explicit weights:

- `.pt`, `.pth`, `.bin`, and `.safetensors` are accepted when the source library can load the model structure.
- URL loading prints the source and raises a clear network/path error on failure.
- Local file loading prints the file path and matched/skipped parameter counts.

## SIE

SIE is only available for supported ViT token backbones. The configured auxiliary SIE is preserved. When `sharing.backbone: true`, a learned modality embedding is automatically added to that SIE; when it is `false`, no modality embedding is added because each modality already owns a private trunk. Do not configure `sie.modality` or `sharing.sie`.

```yaml
model:
  encoder:
    sie:
      enabled: true
      scale: 3.0
      init: "trunc_normal"
      std: 0.02
      dim_index: 0
```

`dim_index` selects exactly one dataset aux dimension. Cardinality comes from real dataset metadata.

CNN final stride defaults to `1`, followed by global pooling. Transformer outputs without a CLS token are globally averaged. ViT/CLIP ViT adapters use CLS; OpenAI CLIP ViT-B/16 has internal width 768 and a 512-dimensional projected/head descriptor, while torchvision ViT-B/16 outputs 768.

## Heads

```yaml
model:
  heads:
    with_normneck: true
    local_global:
      enabled: false
    sharing:
      modal_heads: false
      modal_centers: false
```

The norm neck is fixed to BatchNorm1d. With `with_normneck: true`, metric and center losses use the pre-neck feature, while classifier logits use the post-neck feature. `inference.neck_feat` controls whether retrieval uses the pre-neck or post-neck descriptor.

`local_global.enabled` is only active for ViT-like token backbones.

## Losses

Loss magnitudes are controlled by `model.losses.weights`:

```yaml
weights:
  encoding_ce: 1.0
  encoding_triplet: 1.0
  encoding_center: 0.0005
  fusion_ce: 1.0
  fusion_triplet: 1.0
  fusion_center: 0.0005
  cross_triplet: 0.0
  cross_center: 0.0
  ortho_intra: 0.5
  ortho_inter: 0.5
```

`triplet.name` chooses the metric-learning shape, for example `hard`, `all`, `hard_softmax`, or `all_softmax`.

`ce.name` chooses the classifier loss shape, for example `ce`, `label_smooth`, `arcface`, `cosface`, `am_softmax`, or `sphereface`.

Recovery loss magnitude lives under `model.mv_srbf.recovery.scale`.

## Recovery And Fusion

```yaml
model:
  mv_srbf:
    n_recovery_heads: 8
    fusion_noise_ratio: 0.1
    fusion:
      mode: "single"
    recovery:
      enabled: true
      num_layers: 1
      scale: 1.0
```

Training samples valid missing-modality masks with `0 <= N_mask < M`. The all-visible case is included.
`recovery.num_layers` selects the number of independently parameterized cross-modal recovery blocks and defaults to `1`.

Gaussian feature noise is applied after optional recovery or zero masking. An internal per-modality MLP is then applied before the modal norm neck and inter-modal orthogonality. It preserves the backbone feature width and is not configurable in YAML. The final fusion MLP runs after orthogonality.

## Solver

```yaml
solver:
  optimizer:
    name: "AdamW"
    base_lr: 0.00035
    use_param_groups: true
    module_lrs:
      backbone: 5.0e-6
      encoder: 0.00035
      modal_heads: 0.00035
      recovery: 0.00035
      fusion: 0.00035
      other: 0.00035
  lr_scheduler:
    name: "CosineAnnealingWarmupLR"
    warmup_factor: 0.1
  max_epochs: 100
  warmup_updates: 10
```

`use_param_groups: true` assigns module-wise learning rates. Every mini-batch performs one optimizer update. `warmup_updates` counts successful optimizer updates.

## Inference

```yaml
inference:
  metric: "cos"
  default_num: 50
  ranks: [1, 5, 10]
  remove_same_aux_dims: [0]
  feature_stage: "fusion"
  feature_aggregation: "mean"
  feat_norm: true
  neck_feat: "before"
```

`feature_stage` selects `modal`, `recover`, or `fusion`. For modal/recover, `feature_aggregation` selects `mean`, `concat`, or `fusion`. Modal mean averages available modalities only; modal concat/fusion mean-fills missing slots. Recovery uses all recovered slots. Native fusion reuses the trained fusion module. At the fusion stage, aggregation is ignored. Output widths are `D`, `M*D`, and the fusion-head width respectively.

`remove_same_aux_dims: [0]` is enabled by default. It removes same-identity
gallery samples whose auxiliary dimension 0 matches the query. For most
datasets this is same-camera filtering; for MSVR310, dimension 0 is scene, so
the default removes same-identity same-scene gallery samples. Set `[]` only to
disable this filtering explicitly.

Search backends:

- `torch`
- `faiss` when installed

AQE and re-ranking are optional and run after feature extraction.

## Grad-CAM

`tools/vis_grad_cam.py --stage encoder` explains the selected modality's encoder classifier. `--stage fusion` explains that modality's contribution to the real fusion classifier while the other modality images remain fixed. With `--target_layer auto`, CNNs use the last convolution before global pooling; ViT/CLIP ViT uses the final block's pre-attention `norm1/ln_1`, removes CLS, and reshapes the runtime patch grid. Swin-style channel-last maps are converted to channel-first, and private backbones register the hook on the selected modality's own encoder.

## Logging And Checkpoints

The local JSONL metrics file is always updated in `output_dir`. Checkpoints are saved under `output_dir/checkpoints`.

Common files:

- `latest.pth`
- `best_train_loss.pth`
- optional step snapshots, controlled by `solver.checkpoint.save_snapshots`
