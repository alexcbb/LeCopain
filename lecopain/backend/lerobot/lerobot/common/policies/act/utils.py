import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from typing import Union, Optional, List, Tuple
from PIL import ImageColor
import functools
import os

class LayerScale(nn.Module):
    """Module scaling input by learned scalar.

    Adapted from timm library.
    """

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def soft_to_hard_mask(
    masks: torch.Tensor,
    convert_one_hot: bool = True,
    use_threshold: bool = False,
    threshold: float = 0.5,
):
    """Convert soft to hard masks."""
    # masks: batch [x n_frames] x n_channels x height x width
    assert masks.ndim == 4 or masks.ndim == 5
    min = torch.min(masks)
    max = torch.max(masks)
    if min < 0:
        raise ValueError(f"Minimum mask value should be >=0, but found {min.cpu().numpy()}")
    if max > 1:
        raise ValueError(f"Maximum mask value should be <=1, but found {max.cpu().numpy()}")

    if use_threshold:
        masks = masks > threshold

    if convert_one_hot:
        mask_argmax = torch.argmax(masks, dim=-3)
        masks = F.one_hot(mask_argmax, masks.shape[-3]).to(torch.float32)
        masks = masks.transpose(-1, -2).transpose(-2, -3)  # B, [F,] H, W, C -> B, [F], C, H, W

    return masks

def init_parameters(layers, weight_init: str = "default"):
    assert weight_init in ("default", "he_uniform", "he_normal", "xavier_uniform", "xavier_normal")
    if isinstance(layers, nn.Module):
        layers = [layers]

    for idx, layer in enumerate(layers):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)

        if hasattr(layer, "weight") and layer.weight is not None:
            gain = 1.0
            if isinstance(layers, nn.Sequential):
                if idx < len(layers) - 1:
                    next = layers[idx + 1]
                    if isinstance(next, nn.ReLU):
                        gain = 2**0.5

            if weight_init == "he_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, gain)
            elif weight_init == "he_normal":
                torch.nn.init.kaiming_normal_(layer.weight, gain)
            elif weight_init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight, gain)
            elif weight_init == "xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight, gain)

def get_activation_fn(name_or_instance: Union[str, nn.Module]) -> nn.Module:
    if isinstance(name_or_instance, nn.Module):
        return name_or_instance
    elif isinstance(name_or_instance, str):
        if name_or_instance.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif name_or_instance.lower() == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function {name_or_instance}")
    else:
        raise ValueError(
            f"Unsupported type for activation function: {type(name_or_instance)}. "
            "Can be `str` or `torch.nn.Module`."
        )

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """

    # num_channels: num_groups
    GROUP_NORM_LOOKUP = {
        16: 2,  # -> channels per group: 8
        32: 4,  # -> channels per group: 8
        64: 8,  # -> channels per group: 8
        128: 8,  # -> channels per group: 16
        256: 16,  # -> channels per group: 16
        512: 32,  # -> channels per group: 16
        1024: 32,  # -> channels per group: 32
        2048: 32,  # -> channels per group: 64
    }

    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(
                        GROUP_NORM_LOOKUP[num_channels], num_channels
                    )
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer

def mix_videos_with_masks(
    video: torch.Tensor, masks: torch.Tensor, alpha: float = 0.4
) -> torch.Tensor:
    input_shape = video.shape
    if masks.ndim == 4:
        color_map_shape = masks.shape[1]
    else:
        color_map_shape = masks.shape[2]
    cmap = color_map(color_map_shape)
    video = (video).to(torch.uint8)
    masks = masks.to(bool)
    input_with_masks = torch.stack(
        [
            draw_segmentation_masks_on_image(frame, mask, colors=cmap, alpha=alpha)
            for frame, mask in zip(video, masks)
        ]
    )
    return input_with_masks.reshape(*input_shape)

def unnorm_image(image: torch.Tensor) -> torch.Tensor:
    """Unnormalize image tensor.

    Args:
        image (Tensor): Image tensor of shape (C, H, W).

    Returns:
        Tensor: Unnormalized image tensor of shape (C, H, W).
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(-1, 1, 1)
    return image * std + mean

def draw_segmentation_masks_on_image(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    colors: Optional[
        Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]
    ] = None,
) -> torch.Tensor:
    """
    Draws segmentation masks on given RGB image.

    The values of the input image should be uint8 between 0 and 255.

    Adapted from torchvision.utils.draw_segmentation_masks to run on GPUs if needed.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (color or list of colors, optional): List containing the colors
            of the masks or single color for all masks. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """
    print
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (num_masks, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            (
                "The image and the masks must have the same height and width,"
                + f"but got {masks.shape[-2:]} and {image.shape[-2:]}"
            )
        )

    num_masks = masks.size()[0]
    if colors is not None and num_masks > len(colors):
        raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

    if num_masks == 0:
        print("masks doesn't contain any mask. No mask was drawn", stacklevel=0)
        return image

    if colors is None:

        def generate_color_palette(num_objects: int):
            palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
            return [tuple((i * palette) % 255) for i in range(num_objects)]

        colors = generate_color_palette(num_masks)

    if not isinstance(colors, list):
        colors = [colors]
    if not isinstance(colors[0], (tuple, str)):
        raise ValueError("colors must be a tuple or a string, or a list thereof")
    if isinstance(colors[0], tuple) and len(colors[0]) != 3:
        raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

    out_dtype = torch.uint8

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        colors_.append(torch.tensor(color, dtype=out_dtype, device=image.device))

    img_to_draw = image.detach().clone()
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)

CMAP_STYLE = "tab"

def get_cmap_style() -> str:
    cmap = os.environ.get("VIDEOSAUR_CMAP")
    if cmap is None:
        cmap = CMAP_STYLE

    if cmap not in ("tab", "generated"):
        raise ValueError(f"Invalid color map {cmap}")

    return cmap

@functools.lru_cache
def color_map(N, normalized=False):
    cmap_style = get_cmap_style()
    if cmap_style == "tab" and N <= len(_OUR_TAB_DATA):
        cmap = np.array(_OUR_TAB_DATA[:N], dtype=np.float32)
        if N >= 8:
            # Replace dark gray with a darkish pink, namely the 6th color of Accent
            cmap[7] = (0.94117647058823528, 0.00784313725490196, 0.49803921568627452)
        if N >= 18:
            # Replace light gray with a red-brown, namely the 12th color of Paired
            cmap[17] = (0.69411764705882351, 0.34901960784313724, 0.15686274509803921)
        if not normalized:
            cmap = (cmap * 255).astype(np.uint8)
    else:
        cmap = generate_color_map(N, normalized)

    return [tuple(c) for c in cmap]


_TAB10_DATA = (
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # 1f77b4
    (1.0, 0.4980392156862745, 0.054901960784313725),  # ff7f0e
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # 2ca02c
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # d62728
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # 9467bd
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # 8c564b
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # e377c2
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # 7f7f7f
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # bcbd22
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # 17becf
)

_TAB20_DATA = (
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # 1f77b4
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),  # aec7e8
    (1.0, 0.4980392156862745, 0.054901960784313725),  # ff7f0e
    (1.0, 0.7333333333333333, 0.47058823529411764),  # ffbb78
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # 2ca02c
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),  # 98df8a
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # d62728
    (1.0, 0.596078431372549, 0.5882352941176471),  # ff9896
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # 9467bd
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),  # c5b0d5
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # 8c564b
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),  # c49c94
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # e377c2
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),  # f7b6d2
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # 7f7f7f
    (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),  # c7c7c7
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # bcbd22
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),  # dbdb8d
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # 17becf
    (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),  # 9edae5
)

_TAB20B_DATA = (
    (0.2235294117647059, 0.23137254901960785, 0.4745098039215686),  # 393b79
    (0.3215686274509804, 0.32941176470588235, 0.6392156862745098),  # 5254a3
    (0.4196078431372549, 0.43137254901960786, 0.8117647058823529),  # 6b6ecf
    (0.611764705882353, 0.6196078431372549, 0.8705882352941177),  # 9c9ede
    (0.38823529411764707, 0.4745098039215686, 0.2235294117647059),  # 637939
    (0.5490196078431373, 0.6352941176470588, 0.3215686274509804),  # 8ca252
    (0.7098039215686275, 0.8117647058823529, 0.4196078431372549),  # b5cf6b
    (0.807843137254902, 0.8588235294117647, 0.611764705882353),  # cedb9c
    (0.5490196078431373, 0.42745098039215684, 0.19215686274509805),  # 8c6d31
    (0.7411764705882353, 0.6196078431372549, 0.2235294117647059),  # bd9e39
    (0.9058823529411765, 0.7294117647058823, 0.3215686274509804),  # e7ba52
    (0.9058823529411765, 0.796078431372549, 0.5803921568627451),  # e7cb94
    (0.5176470588235295, 0.23529411764705882, 0.2235294117647059),  # 843c39
    (0.6784313725490196, 0.28627450980392155, 0.2901960784313726),  # ad494a
    (0.8392156862745098, 0.3803921568627451, 0.4196078431372549),  # d6616b
    (0.9058823529411765, 0.5882352941176471, 0.611764705882353),  # e7969c
    (0.4823529411764706, 0.2549019607843137, 0.45098039215686275),  # 7b4173
    (0.6470588235294118, 0.3176470588235294, 0.5803921568627451),  # a55194
    (0.807843137254902, 0.42745098039215684, 0.7411764705882353),  # ce6dbd
    (0.8705882352941177, 0.6196078431372549, 0.8392156862745098),  # de9ed6
)

# This colormap first contains the tab10 colors, then every second color of the tab20 colors, and
# then the colors of tab20b
_OUR_TAB_DATA = (
    _TAB10_DATA
    + _TAB20_DATA[1::2]
    + _TAB20B_DATA[::4]
    + _TAB20B_DATA[1::4]
    + _TAB20B_DATA[2::4]
    + _TAB20B_DATA[3::4]
)

def generate_color_map(N, normalized=False):
    dtype = np.float32 if normalized else np.uint8

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = (r, g, b)

    cmap = cmap / 255 if normalized else cmap

    return cmap