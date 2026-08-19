import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms import functional as F
from typing import Tuple, Dict, Any, Optional, Union
from PIL import Image


_INTERP_MAP = {
    "nearest": F.InterpolationMode.NEAREST,
    "bilinear": F.InterpolationMode.BILINEAR,
    "bicubic": F.InterpolationMode.BICUBIC,
}

class MultiSpectralTransform:
    def __init__(
        self,
        input_size: Tuple[int, int] = (128, 256),
        interpolation: str = "bilinear",
        normalize_params: Optional[Dict[str, Dict[str, list]]] = None,
        shared_aug: Optional[Dict[str, Any]] = None,
        padding: Tuple[int, int] = (0, 0),
        padding_fill: Union[int, Tuple[int, int, int]] = 0,
    ):
        self.input_size = input_size
        self.interpolation = _INTERP_MAP[str(interpolation).lower()]
        self.normalize_params = normalize_params or {}
        self.shared_aug = shared_aug or {}
        self.padding = (int(padding[0]), int(padding[1]))
        if self.padding[0] < 0 or self.padding[1] < 0:
            raise ValueError(f"padding values must be non-negative, got {self.padding}.")
        self.padding_fill = padding_fill
        blocked = {
            "random" + "_rotation_degrees",
            "random" + "_resized_crop",
            "per" + "_modal",
        }
        present = sorted(k for k in blocked if k in self.shared_aug)
        if present:
            raise ValueError(f"Unsupported augmentation field(s): {present}.")

        self.initial_resize = v2.Resize(input_size, interpolation=self.interpolation)

        shared_ops = []
        if "random_horizontal_flip" in self.shared_aug:
            p = self.shared_aug["random_horizontal_flip"]
            shared_ops.append(v2.RandomHorizontalFlip(p=p))

        self.shared_transform = v2.Compose(shared_ops) if shared_ops else None
        self.post_pad_transform = self._build_post_pad_transform()

    def _build_post_pad_transform(self) -> Optional[v2.Compose]:
        crop_cfg = self.shared_aug.get("random_crop")
        if not crop_cfg:
            return None
        if isinstance(crop_cfg, dict):
            size = tuple(crop_cfg.get("size", self.input_size))
            padding = crop_cfg.get("padding", None)
            pad_if_needed = bool(crop_cfg.get("pad_if_needed", False))
        else:
            size = self.input_size
            padding = None
            pad_if_needed = False
        return v2.Compose(
            [
                v2.RandomCrop(
                    size=size,
                    padding=padding,
                    pad_if_needed=pad_if_needed,
                    fill=self.padding_fill,
                )
            ]
        )

    def _pad_image(self, img: Image.Image) -> Image.Image:
        """Apply global symmetric input padding after resize/geometric transforms."""
        pad_h, pad_w = self.padding
        if pad_h == 0 and pad_w == 0:
            return img
        return F.pad(img, [pad_w, pad_h, pad_w, pad_h], fill=self.padding_fill)

    def _get_normalizer(self, modal_name: str) -> v2.Normalize:
        if modal_name in self.normalize_params:
            mean = self.normalize_params[modal_name]["mean"]
            std = self.normalize_params[modal_name]["std"]
        elif "default" in self.normalize_params:
            mean = self.normalize_params["default"]["mean"]
            std = self.normalize_params["default"]["std"]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        return v2.Normalize(mean=mean, std=std)

    def __call__(self, modalities: Dict[str, Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Args:
            modalities: dict like {"modal1": rgb_img, "modal2": nir_img, "modal3": tir_img}
                        Only modalities present will be processed.

        Returns:
            dict of normalized tensors with same keys.
        """
        if not modalities:
            raise ValueError("Input modalities dict is empty.")

        keys = list(modalities.keys())
        pil_list = [modalities[k] for k in keys]
        resized_pils = [self.initial_resize(img) for img in pil_list]

        if self.shared_transform is None:
            transformed_pils = resized_pils
        else:
            transformed_pils = self.shared_transform(resized_pils)
        transformed_pils = [self._pad_image(img) for img in transformed_pils]
        if self.post_pad_transform is not None:
            transformed_pils = self.post_pad_transform(transformed_pils)

        output = {}
        for i, key in enumerate(keys):
            img = transformed_pils[i]

            tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img)

            normalizer = self._get_normalizer(key)
            tensor = normalizer(tensor)

            output[key] = tensor

        return output
