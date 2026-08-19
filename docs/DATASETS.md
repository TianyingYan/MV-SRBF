# Datasets

MV-SRBF supports vehicle ReID datasets in this release.

Supported dataset names:

- `MSVR310`
- `RGBNT100`
- `RGBN300`
- `WMVEID863`

Supported dataset YAML files:

- `configs/data/msvr310.yaml`
- `configs/data/rgbnt100.yaml`
- `configs/data/rgbn300.yaml`
- `configs/data/wmveid863.yaml`

`dataset.num_classes` is inferred from the real training split before model construction, so the YAML value is only a fallback.

The default `dataset.root` is `../datasets`. Relative roots use the working
directory captured when the configuration is loaded and do not depend on the
location of `data/dataloaders.py`. With commands launched from the
MV-SRBF repository root, the default resolves to the sibling `datasets`
directory. Absolute roots are also supported.

## Modalities

The dataset reader infers modality keys from the sample metadata. The model then builds the modality list from the merged train/query/gallery metadata.

Common keys are:

- `modal1`
- `modal2`
- `modal3`

The display names in YAML are only labels used in logs and visualization.

## Input Size

`input.size: "auto"` resolves to `[128, 256]`, the vehicle ReID image geometry used by the supported configs.

The preprocessing order is:

1. resize
2. shared horizontal flip during training
3. global symmetric padding
4. shared random crop during training
5. tensor conversion
6. normalization

## Aux Dimensions

Dataset tuple fields after identity are treated as auxiliary metadata. The dataloader compacts every observed aux value into contiguous indices and exposes the mapping to the model.

Typical use:

- `dataloader.train.sampler.aux_dims` controls identity sampler grouping.
- `model.encoder.sie.dim_index` controls the single aux dimension used by SIE.
- `inference.remove_same_aux_dims` removes same-ID gallery images that also match selected aux dimensions.

The default is `remove_same_aux_dims: [0]`. Auxiliary dimension 0 is camera for
most datasets and scene for MSVR310, so the default performs same-ID
same-camera filtering or same-ID same-scene filtering respectively. Use `[]`
only when this filtering must be disabled.
