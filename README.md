# MV-SRBF

MV-SRBF is a vehicle-oriented multi-modal ReID project. This release focuses on four supported datasets:

- `MSVR310`
- `RGBNT100`
- `RGBN300`
- `WMVEID863`

The code supports multi-modal vehicle training, optional mask recovery, fusion, AQE/re-ranking, any-modal evaluation, and retrieval/feature visualization.

## Installation

Run installation from the Python environment selected for your platform:

```console
python -m pip install -e .
```

OpenAI CLIP support requires the official package:

```console
python -m pip install git+https://github.com/openai/CLIP.git
```

Optional visualization extras:

```console
python -m pip install -e ".[visualization]"
```

## Configuration

Experiment YAML files live in `configs/experiments/`, and dataset YAML files live in `configs/data/`.

Supported experiment entries:

- `configs/experiments/msvr310.yaml`
- `configs/experiments/rgbnt100.yaml`
- `configs/experiments/rgbn300.yaml`
- `configs/experiments/wmveid863.yaml`

`input.size: "auto"` resolves to the vehicle ReID geometry `[128, 256]`.

Training augmentation is intentionally compact:

- `random_horizontal_flip`
- `padding`
- `random_crop`
- batch random erasing with `mode: synchronous` or `asynchronous`

## Backbone And Heads

Backbones are configured under `model.encoder.backbone`. The default release config uses OpenAI CLIP ViT-B/16 with local weights:

```yaml
source: "openai-clip"
name: "ViT-B/16"
pretrained: true
pretrained_weights_path: "../weights/clip_vit_b_16.pt"
```

The ReID head uses `with_normneck` and a fixed BatchNorm neck. ID loss is applied after the neck classifier, while metric and center losses use the pre-neck feature.

SIE preserves the selected auxiliary dimension. With a shared backbone it automatically adds modality identity; with private backbones it does not:

```yaml
sie:
  enabled: true
  dim_index: 0
```

Aux values are inferred from the real dataset metadata and compacted automatically.

Relative runtime paths are resolved from the working directory in which the
configuration is loaded, not from the installed package location. The shipped
defaults assume commands are launched from the MV-SRBF repository root and
use the sibling paths `../datasets` and
`../weights/clip_vit_b_16.pt`. Absolute paths remain supported.

## Training

```console
python tools/train.py --config configs/experiments/msvr310.yaml
```

Useful options:

- `--parameter key=value` overrides YAML fields from the command line.
- `--breakpoint PATH` resumes from a project checkpoint.
- `--no-memory` loads project weights but resets optimizer, scheduler, epoch, and early-stop state.
- `--force-load` skips incompatible tensors when loading project weights.

The local JSONL log is always written under the configured `output_dir`.

## Evaluation

```console
python tools/infer.py --config configs/experiments/msvr310.yaml
```

By default, inference loads `best_train_loss.pth` from the run checkpoint directory if it exists. If no project checkpoint is present, the initialized model is used, including configured backbone pretrained weights.

Evaluation supports:

- cosine or L2 distance
- `feat_norm`
- `neck_feat: before | after`
- `feature_stage: modal | recover | fusion`
- `feature_aggregation: mean | concat | fusion` for modal/recover; native fusion is always used at the fusion stage
- AQE
- k-reciprocal re-ranking
- fixed any-modal symmetric report
- `remove_same_aux_dims: [0]` by default, removing same-ID matches on auxiliary
  dimension 0 (camera for most datasets, scene for MSVR310)

## Visualization

Top-K retrieval visualization:

```console
python tools/vis_any_modal_topk.py --config configs/experiments/msvr310.yaml --out_dir logs/msvr310/topk
```

t-SNE:

```console
python tools/vis_tsne.py --config configs/experiments/msvr310.yaml --out logs/msvr310/tsne.png
```

Feature PDF/CDF:

```console
python tools/vis_feature_distribution.py --config configs/experiments/msvr310.yaml --out logs/msvr310/feature_dist.png
```

Grad-CAM uses `pytorch-grad-cam` when installed:

```console
python tools/vis_grad_cam.py --config configs/experiments/msvr310.yaml --out_dir logs/msvr310/grad_cam --stage fusion
```

## More Docs

- `docs/CONFIGURATION.md`
- `docs/DATASETS.md`
- `docs/PRETRAINED_WEIGHTS.md`

## License

MV-SRBF is released under the [MIT License](LICENSE).
