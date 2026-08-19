# Pretrained Weights

MV-SRBF supports backbone initialization from the source library or from an explicit local file.

## Local Weight Files

The shipped defaults assume commands are launched from the MV-SRBF
repository root and place local pretrained weights in its sibling directory:

```text
../weights/
```

The default OpenAI CLIP ViT-B/16 config uses:

```yaml
model:
  encoder:
    backbone:
      source: "openai-clip"
      name: "ViT-B/16"
      pretrained: true
      pretrained_weights_path: "../weights/clip_vit_b_16.pt"
```

Supported local file suffixes depend on the selected source backend, but the project accepts common PyTorch and safetensors files such as `.pt`, `.pth`, `.bin`, and `.safetensors` when the source adapter can match the tensors.

## Source Backends

- `openai-clip`: uses the official OpenAI CLIP package installed from `git+https://github.com/openai/CLIP.git`.
- `open-clip`: uses `open_clip_torch`.
- `torchvision`: uses `torchvision.models`.
- `timm`: uses `timm.create_model`.
- `huggingface-hub` / `transformers`: uses HuggingFace model loaders.
- `custom`: uses a user-provided Python import target.

Relative `pretrained_weights_path` values use the same working-directory base as
`dataset.root`, while absolute paths are used unchanged. When the field is set,
source-library download is skipped and the resolved local file is loaded
instead.

## Project Checkpoints

Project checkpoints such as `latest.pth` and `best_train_loss.pth` are not backbone pretrained weights. They are loaded by training resume or inference tools.

During inference and breakpoint resume, `model.encoder.backbone.pretrained` and `pretrained_weights_path` are ignored because the project checkpoint supplies the model weights.
