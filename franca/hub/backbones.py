import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from franca.hub.utils import _TEMPDIR, extract_tar_file, load_state_dict_from_url

_FRANCA_BASE_URL = "https://github.com/valeoai/Franca/releases/download/v1.0.0"
_FRANCA_ViT_G_CHUNKS = [
    "_chunked.tar.gz.part_aa",
    "_chunked.tar.gz.part_ab",
    "_chunked.tar.gz.part_ac",
]


def _make_franca_model_name(arch_name: str, patch_size: int, pretraining_dataset: str) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"franca_{compact_arch_name}{patch_size}_{pretraining_dataset}"


@dataclass
class FrancaConfig:
    arch: str = "vit_large"
    patch_size: int = 14
    layerscale: float = 1.0e-05
    ffn_layer: str = "swiglufused"
    block_chunks: int = 4
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    num_register_tokens: int = 0
    interpolate_antialias: bool = False
    interpolate_offset: float = 0.1


class Weights(Enum):
    IN21K = "In21K"
    LAION = "Laion600M"
    DINOV2_IN21K = "Dinov2_In21K"


def _make_franca_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IN21K,
    local_state_dict: Optional[str | list[str]] = None,
    **kwargs,
) -> nn.Module:
    from ..models import build_model

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    vit_config = FrancaConfig(arch=arch_name, **kwargs)
    model, _ = build_model(vit_config, only_teacher=True, img_size=img_size)

    model_full_name = _make_franca_model_name(arch_name, vit_config.patch_size, weights.value)

    if pretrained:
        if weights == Weights.LAION and arch_name == "vit_base":
            raise ValueError(
                "Franca ViT-B/14 model is not available with LAION weights. "
                "Please use IN21K weights or set `pretrained=False`."
            )
        if local_state_dict is not None:
            # This is mainly for testing purposes
            if os.path.isdir(local_state_dict):
                with tempfile.TemporaryDirectory(dir=_TEMPDIR) as tmpdirname:
                    outfile = extract_tar_file(local_state_dict, tmpdirname)
                    state_dict = torch.load(os.path.join(tmpdirname, outfile), map_location="cpu", weights_only=True)
            else:
                state_dict = torch.load(local_state_dict, map_location="cpu", weights_only=True)
        else:
            if arch_name == "vit_giant2":
                url = [_FRANCA_BASE_URL + f"/{chunk}" for chunk in _FRANCA_ViT_G_CHUNKS]
            else:
                url = _FRANCA_BASE_URL + f"/{model_full_name}.pth"
            state_dict = load_state_dict_from_url(url, map_location="cpu", weights_only=True)
        state_dict: dict[str, Any] = state_dict["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)

        if len(msg.missing_keys) != 0:
            raise ValueError(
                f"Missing keys in the state_dict: {msg.missing_keys}. "
                "Ensure that the model architecture matches the state_dict."
            )

        for k in msg.unexpected_keys:
            if k.startswith("dino_head.") or k.startswith("ibot_head."):
                continue
            raise ValueError(
                f"Unexpected key in the state_dict: {k}. Ensure that the model architecture matches the state_dict."
            )

        flat_blocks = []
        for chunk in model.blocks:  # chunk is a BlockChunk (nn.ModuleList)
            for blk in chunk:  # blk is either Identity() or a NestedTensorBlock
                if not isinstance(blk, nn.Identity):
                    flat_blocks.append(blk)

        # replace model.blocks with the flat list…
        model.blocks = nn.ModuleList(flat_blocks)
        model.chunked_blocks = False  # so the forward logic uses the "not chunked" path

        assert len(model.blocks) == model.n_blocks, f"Expected {model.n_blocks} blocks, but got {len(model.blocks)} blocks."

    return model


def franca_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.IN21K, **kwargs) -> nn.Module:
    """
    Franca ViT-B/14 model (optionally) pretrained on the In21K dataset.
    """
    if weights == Weights.DINOV2_IN21K or "Dinov2_In21k":
        img_size = kwargs.pop("img_size", 224)
    else:
        img_size = kwargs.pop("img_size", 518)
    return _make_franca_model(arch_name="vit_base", pretrained=pretrained, weights=weights, img_size=img_size, **kwargs)


def franca_vitl14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.IN21K, **kwargs) -> nn.Module:
    """
    Franca ViT-L/14 model (optionally) pretrained on either the In21K or Laion600M dataset.
    """
    return _make_franca_model(arch_name="vit_large", pretrained=pretrained, weights=weights, **kwargs)


def franca_vitg14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.IN21K, **kwargs) -> nn.Module:
    """
    Franca ViT-g/14 model (optionally) pretrained on either the In21K or Laion600M dataset.
    """
    return _make_franca_model(arch_name="vit_giant2", weights=weights, pretrained=pretrained, **kwargs)
