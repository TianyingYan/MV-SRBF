# Requirements

This project is distributed as a normal Python package. Run commands from the Python environment selected for your platform, and use `python -m pip` so packages are installed into that interpreter.

## Install

Core install:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` is the reproducible dependency set for the current validated runtime environment and project extras. Runtime packages are version-pinned except the official OpenAI CLIP package, which is installed from `git+https://github.com/openai/CLIP.git`. PyTorch wheels are platform and CUDA specific. The validated environment uses `torch 2.6.0+cu124` and `torchvision 0.21.0+cu124`; install those matching CUDA 12.4 wheels from the PyTorch index first if the active package index does not provide them, then install the remaining requirements.

Optional extras:

```bash
python -m pip install -e ".[faiss]"
python -m pip install -e ".[logging]"
python -m pip install -e ".[timm]"
python -m pip install -e ".[openai-clip]"
python -m pip install -e ".[open-clip]"
python -m pip install -e ".[transformers]"
python -m pip install -e ".[flops]"
python -m pip install -e ".[visualization]"
python -m pip install -e ".[all]"
```

- `faiss`: enables `inference.search.backend: faiss`.
- `logging`: installs `trackio` for experiment tracking and the local web dashboard (`logging.tracker: trackio`). The code falls back to local JSONL logs when the tracker is unavailable or set to `none`.
- `timm`: enables `model.encoder.backbone.source: timm`.
- `openai-clip`: installs the official OpenAI `clip` package from `git+https://github.com/openai/CLIP.git` for `model.encoder.backbone.source: openai-clip`. Do not use the PyPI `openai-clip` package because it carries obsolete Torch constraints.
- `open-clip`: installs `open_clip_torch` for `model.encoder.backbone.source: open-clip`.
- `transformers`: installs HuggingFace `transformers` for `model.encoder.backbone.source: huggingface` / `transformers` / `hf`.
- `flops`: installs `fvcore` for publication-table FLOPs. Without it, the trainer can fall back to the lightweight hook estimate when `logging.model_summary.allow_fallback: true`.
- `visualization`: installs matplotlib, pytorch-grad-cam, and scikit-learn for Grad-CAM, t-SNE, and PDF/CDF plots.
- `all`: installs all optional runtime features.

## Current Validation Environment

The dependency versions can be inspected with:

```console
python -c "import sys, torch, torchvision, numpy, PIL, yaml, tifffile, safetensors; print(sys.version, torch.__version__, torchvision.__version__, numpy.__version__, PIL.__version__, yaml.__version__, tifffile.__version__, safetensors.__version__)"
```

| Package | Version |
| --- | --- |
| Python | 3.13.13 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| CUDA runtime | 12.4 |
| numpy | 2.4.6 |
| Pillow | 12.2.0 |
| PyYAML | 6.0.3 |
| tifffile | 2026.6.1 |
| safetensors | 0.8.0 |
| OpenAI CLIP | official GitHub package from `git+https://github.com/openai/CLIP.git`; provides the `clip` import used by `model.encoder.backbone.source: openai-clip` |
| matplotlib | 3.11.0 via `.[visualization]` for plots |
| grad-cam | 1.5.5 via `.[visualization]` for `pytorch_grad_cam` |
| scikit-learn | 1.9.0 via `.[visualization]` for t-SNE |
| timm | 1.0.27 via `.[timm]` for `model.encoder.backbone.source: timm` |
| open_clip_torch | 3.3.0 via `.[open-clip]` for `model.encoder.backbone.source: open-clip` |
| transformers | 5.12.0 via `.[transformers]` for HuggingFace backbones |
| fvcore | 0.1.5.post20221221 via `.[flops]` for the FLOPs table |
| faiss | 1.14.3 via `.[faiss]` for the faiss search backend |
| trackio | 0.27.0 via `.[logging]` for tracking + the local web dashboard (`logging.tracker: trackio`) |
| gradio | not required for core MV-SRBF; Trackio manages its own dashboard dependencies |

This environment runs **numpy 2.x on Python 3.13 with a CUDA 12.4 build of PyTorch**. Direct dependencies are pinned to the versions above in `requirements.txt` and `pyproject.toml`, except the official OpenAI CLIP GitHub dependency; transitive packages remain managed by pip.

> Note: the `trackio` dashboard is optional. If `pip install` falls back to building an optional package from source and fails on Python 3.13, install a version that ships a cp313 wheel or skip that extra; none are needed for core training/inference with the default `openai-clip` backbone.
