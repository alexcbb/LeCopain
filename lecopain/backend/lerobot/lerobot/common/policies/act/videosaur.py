import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import copy
import timm
import math
from typing import Dict, Optional, Tuple

import lerobot.common.policies.act.utils as utils
from typing import Optional, List, Union, Dict, Any, Tuple, Callable
import torchvision

class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        initial_layer_norm: bool = False,
        activation: Union[str, nn.Module] = "relu",
        final_activation: Union[bool, str] = False,
        residual: bool = False,
        weight_init: str = "default",
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert inp_dim == outp_dim

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, dim))
            layers.append(utils.get_activation_fn(activation))
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))
        if final_activation:
            if isinstance(final_activation, bool):
                final_activation = "relu"
            layers.append(utils.get_activation_fn(final_activation))

        self.layers = nn.Sequential(*layers)
        utils.init_parameters(self.layers, weight_init)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)

        if self.residual:
            return inp + outp
        else:
            return outp

class SlotAttention(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
    ):
        super().__init__()
        assert n_iters >= 1

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim

        if use_mlp:
            self.mlp = MLP(
                slot_dim, slot_dim, [hidden_dim], initial_layer_norm=True, residual=True
            )
        else:
            self.mlp = None

        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim**-0.5

    def step(
        self, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one iteration of slot attention."""
        slots = self.norm_slots(slots)
        queries = self.to_q(slots)

        dots = torch.einsum("bsd, bfd -> bsf", queries, keys) * self.scale
        pre_norm_attn = torch.softmax(dots, dim=1)
        attn = pre_norm_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        if self.gru:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn

    def forward(self, slots: torch.Tensor, features: torch.Tensor, n_iters: Optional[int] = None):
        features = self.norm_features(features)
        keys = self.to_k(features)
        values = self.to_v(features)

        if n_iters is None:
            n_iters = self.n_iters

        for _ in range(n_iters):
            slots, pre_norm_attn = self.step(slots, keys, values)

        return {"slots": slots, "masks": pre_norm_attn}

class RandomInit(nn.Module):
    """Sampled random initialization for all slots."""

    def __init__(self, n_slots: int, dim: int, initial_std: Optional[float] = None):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(1, 1, dim))
        if initial_std is None:
            initial_std = dim**-0.5
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, dim) * initial_std))

    def forward(self, batch_size: int):
        noise = torch.randn(batch_size, self.n_slots, self.dim, device=self.mean.device)
        return self.mean + noise * self.log_std.exp()

class Attention(nn.Module):
    """Multihead attention.

    Adapted from timm's ViT implementation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        inner_dim: Optional[int] = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kdim = dim if kdim is None else kdim
        vdim = dim if vdim is None else vdim
        inner_dim = dim if inner_dim is None else inner_dim
        if inner_dim % num_heads != 0:
            raise ValueError("`inner_dim` must be divisible by `num_heads`")

        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim**-0.5

        self._same_qkv_dim = dim == kdim and dim == vdim
        self._same_kv_dim = kdim == vdim

        if self._same_qkv_dim:
            self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        elif self._same_kv_dim:
            self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
            self.kv = nn.Linear(kdim, inner_dim * 2, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
            self.k = nn.Linear(kdim, inner_dim, bias=qkv_bias)
            self.v = nn.Linear(vdim, inner_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.out_proj_drop = nn.Dropout(proj_drop)

        self.init_parameters()

    def init_parameters(self):
        if self._same_qkv_dim:
            bound = math.sqrt(6.0 / (self.qkv.weight.shape[0] // 3 + self.qkv.weight.shape[1]))
            nn.init.uniform_(self.qkv.weight, -bound, bound)  # Xavier init for separate Q, K, V
            if self.qkv.bias is not None:
                nn.init.constant_(self.qkv.bias, 0.0)
        elif self._same_kv_dim:
            utils.init_parameters(self.q, "xavier_uniform")
            bound = math.sqrt(6.0 / (self.kv.weight.shape[0] // 2 + self.kv.weight.shape[1]))
            nn.init.uniform_(self.kv.weight, -bound, bound)  # Xavier init for separate K, V
            if self.kv.bias is not None:
                nn.init.constant_(self.kv.bias, 0.0)
        else:
            utils.init_parameters((self.q, self.k, self.v), "xavier_uniform")

        utils.init_parameters(self.out_proj, "xavier_uniform")

    def _in_proj(self, q, k, v):
        """Efficiently compute in-projection.

        Adapted from torch.nn.functional.multi_head_attention.
        """
        if self._same_qkv_dim:
            w_kv = b_kv = b_q = b_k = b_v = None
            w = self.qkv.weight
            b = self.qkv.bias if hasattr(self.qkv, "bias") else None
        elif self._same_kv_dim:
            w = b = b_k = b_v = None
            w_q = self.q.weight
            w_kv = self.kv.weight
            b_q = self.q.bias if hasattr(self.q, "bias") else None
            b_kv = self.kv.bias if hasattr(self.kv, "bias") else None
        else:
            w = w_kv = b = b_kv = None
            w_q = self.q.weight
            w_k = self.k.weight
            w_v = self.v.weight
            b_q = self.q.bias if hasattr(self.q, "bias") else None
            b_k = self.k.bias if hasattr(self.k, "bias") else None
            b_v = self.v.bias if hasattr(self.v, "bias") else None

        if k is v:
            if q is k:
                # Self-attention
                return F.linear(q, w, b).chunk(3, dim=-1)
            else:
                # Encoder-decoder attention
                if w is not None:
                    dim = w.shape[0] // 3
                    w_q, w_kv = w.split([dim, dim * 2])
                    if b is not None:
                        b_q, b_kv = b.split([dim, dim * 2])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            if w is not None:
                w_q, w_k, w_v = w.chunk(3)
                if b is not None:
                    b_q, b_k, b_v = b.chunk(3)
            elif w_kv is not None:
                w_k, w_v = w_kv.chunk(2)
                if b_kv is not None:
                    b_k, b_v = b_kv.chunk(2)

            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = key if key is not None else query
        value = value if value is not None else query

        bs, n_queries, _ = query.shape
        n_keys = key.shape[1]

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                expected = (n_queries, n_keys)
                if attn_mask.shape != expected:
                    raise ValueError(
                        f"2D `attn_mask` should have shape {expected}, but has "
                        f"shape {attn_mask.shape}"
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.ndim == 3:
                expected = (bs * self.num_heads, n_queries, n_keys)
                if attn_mask.shape != expected:
                    raise ValueError(
                        f"3D `attn_mask` should have shape {expected}, but has "
                        f"shape {attn_mask.shape}"
                    )
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool
            expected = (bs, n_keys)
            if key_padding_mask.shape != expected:
                raise ValueError(
                    f"`key_padding_mask` should have shape {expected}, but has shape "
                    f"{key_padding_mask.shape}"
                )
            key_padding_mask = einops.repeat(
                key_padding_mask, "b n -> (b h) 1 n", b=bs, h=self.num_heads, n=n_keys
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        q, k, v = self._in_proj(query, key, value)

        q = einops.rearrange(q, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)
        k = einops.rearrange(k, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)
        v = einops.rearrange(v, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)

        q_scaled = q / self.scale
        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q_scaled, k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)  # (B x H) x N x M
        pre_dropout_attn = attn
        attn = self.attn_drop(attn)

        weighted_v = attn @ v
        x = einops.rearrange(weighted_v, "(b h) n d -> b n (h d)", h=self.num_heads, d=self.head_dim)
        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        if return_weights:
            weights = einops.rearrange(pre_dropout_attn, "(b h) n m -> b h n m", h=self.num_heads)
            return x, weights.mean(dim=1)
        else:
            return x, None

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Like torch.nn.TransformerEncoderLayer, but with customizations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dim_attn: Optional[int] = None,
        dim_kv: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        initial_residual_scale: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device=device,
            dtype=dtype,
        )
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            kdim=dim_kv,
            vdim=dim_kv,
            inner_dim=dim_attn,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        if initial_residual_scale is not None:
            self.scale1 = utils.LayerScale(d_model, init_values=initial_residual_scale)
            self.scale2 = utils.LayerScale(d_model, init_values=initial_residual_scale)
        else:
            self.scale1 = nn.Identity()
            self.scale2 = nn.Identity()

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        keys = keys if keys is not None else x
        values = values if values is not None else x
        x, attn = self.self_attn(
            x,
            keys,
            values,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_weights=return_weights,
        )
        x = self.dropout1(x)

        if return_weights:
            return x, attn
        else:
            return x

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self.scale1(
                self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, keys=memory, values=memory
                )
            )
            x = x + self.scale2(self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(
                x
                + self.scale1(
                    self._sa_block(x, src_mask, src_key_padding_mask, keys=memory, values=memory)
                )
            )
            x = self.norm2(x + self.scale2(self._ff_block(x)))

        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_blocks: int,
        n_heads: int,
        qkv_dim: Optional[int] = None,
        memory_dim: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        hidden_dim: Optional[int] = None,
        initial_residual_scale: Optional[float] = None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim,
                    n_heads,
                    dim_feedforward=hidden_dim,
                    dim_attn=qkv_dim,
                    dim_kv=memory_dim,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=True,
                    norm_first=True,
                    initial_residual_scale=initial_residual_scale,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        inp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = inp

        for block in self.blocks:
            x = block(x, mask, key_padding_mask, memory)

        return x

class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{i + 1}": f"blocks.{i}" for i in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")
        
        model = TimmExtractor._create_model(model_name, pretrained, checkpoint_path, model_kwargs)

        if self.features is not None:
            nodes = torchvision.models.feature_extraction.get_graph_node_names(model)[0]

            features = []
            for name in self.features:
                if name in TimmExtractor.FEATURE_ALIASES:
                    name = TimmExtractor.FEATURE_ALIASES[name]

                if not any(node.startswith(name) for node in nodes):
                    raise ValueError(
                        f"Requested features under node {name}, but this node does "
                        f"not exist in model {model_name}. Available nodes: {nodes}"
                    )

                features.append(name)

            model = torchvision.models.feature_extraction.create_feature_extractor(model, features)

        self.model = model

        if self.frozen:
            self.requires_grad_(False)

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractor._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            )

        return model

    def forward(self, inp):
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        if self.features is not None:
            if self.is_vit:
                outputs = {k: v[:, 1:] for k, v in outputs.items()}  # Remove CLS token
            outputs = {self.FEATURE_MAPPING[key]: value for key, value in outputs.items()}
            for name in self.features:
                if ("keys" in name) or ("queries" in name) or ("values" in name):
                    feature_name = name.replace("queries", "keys").replace("values", "keys")
                    B, N, C = outputs[feature_name].shape
                    qkv = outputs[feature_name].reshape(
                        B, N, 3, C // 3
                    )  # outp has shape B, N, 3 * H * (C // H)
                    q, k, v = qkv.unbind(2)
                    if "keys" in name:
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v
                    else:
                        raise ValueError(f"Unknown feature name {name}.")

            if len(outputs) == 1:
                # Unpack single output for now
                return next(iter(outputs.values()))
            else:
                return outputs
        else:
            return outputs

class DinoEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
        model: str = "vit_base_patch8_224",
        output_transform: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = TimmExtractor(
            model=model,
            pretrained=True,
            features=["vit_block12", "vit_block_keys12"],
            frozen=True,
        )
        self.output_transform = output_transform
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key

    def forward(
            self, 
            images,
            train=True
        ):
        # images: batch x n_channels x height x width
        backbone_features = self.backbone(images)
        if isinstance(backbone_features, dict):
            features = backbone_features[self.main_features_key].clone()
        else:
            features = backbone_features.clone()
        
        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if self.output_transform:
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            main_backbone_features = backbone_features[self.main_features_key]

            return {
                "features": features,
                "backbone_features": main_backbone_features,
                **backbone_features,
            }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"

            return {
                "features": features,
                "backbone_features": backbone_features,
            }

class SlotMixerDecoder(nn.Module):
    """Slot mixer decoder reconstructing jointly over all slots, but independent per position.

    Introduced in Sajjadi et al., 2022: Object Scene Representation Transformer,
    http://arxiv.org/abs/2206.06922
    """

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        embed_dim: int,
        n_patches: int,
        allocator: nn.Module,
        renderer: nn.Module,
        renderer_dim: Optional[int] = None,
        output_transform: Optional[nn.Module] = None,
        pos_embed_mode: Optional[str] = None,
        use_layer_norms: bool = False,
        norm_memory: bool = True,
        temperature: Optional[float] = None,
        eval_output_size: Optional[Tuple[int]] = None,
    ):
        super().__init__()
        self.allocator = allocator
        self.renderer = renderer
        self.eval_output_size = list(eval_output_size) if eval_output_size else None

        att_dim = max(embed_dim, inp_dim)
        self.scale = att_dim**-0.5 if temperature is None else temperature**-1
        self.to_q = nn.Linear(embed_dim, att_dim, bias=False)
        self.to_k = nn.Linear(inp_dim, att_dim, bias=False)

        if use_layer_norms:
            self.norm_k = nn.LayerNorm(inp_dim, eps=1e-5)
            self.norm_q = nn.LayerNorm(embed_dim, eps=1e-5)
            self.norm_memory = norm_memory
            if norm_memory:
                self.norm_memory = nn.LayerNorm(inp_dim, eps=1e-5)
            else:
                self.norm_memory = nn.Identity()
        else:
            self.norm_k = nn.Identity()
            self.norm_q = nn.Identity()
            self.norm_memory = nn.Identity()

        if output_transform is None:
            if renderer_dim is None:
                raise ValueError("Need to provide render_mlp_dim if output_transform is unspecified")
            self.output_transform = nn.Linear(renderer_dim, outp_dim)
        else:
            self.output_transform = output_transform

        if pos_embed_mode is not None and pos_embed_mode not in ("none", "add", "concat"):
            raise ValueError("If set, `pos_embed_mode` should be 'none', 'add' or 'concat'")
        self.pos_embed_mode = pos_embed_mode
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, embed_dim) * embed_dim**-0.5)
        self.init_parameters()

    def init_parameters(self):
        layers = [self.to_q, self.to_k]
        if isinstance(self.output_transform, nn.Linear):
            layers.append(self.output_transform)
        utils.init_parameters(layers, "xavier_uniform")

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.training and self.eval_output_size is not None:
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(
                self.pos_emb,
                new_size=self.eval_output_size,
                num_prefix_tokens=0,
            )
        else:
            pos_emb = self.pos_emb

        pos_emb = pos_emb.expand(len(slots), -1, -1)
        memory = self.norm_memory(slots)
        query_features = self.allocator(pos_emb, memory=memory)
        
        q = self.to_q(self.norm_q(query_features))  # B x P x D
        k = self.to_k(self.norm_k(slots))  # B x S x D

        dots = torch.einsum("bpd, bsd -> bps", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        mixed_slots = torch.einsum("bps, bsd -> bpd", attn, slots)  # B x P x D

        if self.pos_embed_mode == "add":
            mixed_slots = mixed_slots + pos_emb
        elif self.pos_embed_mode == "concat":
            mixed_slots = torch.cat((mixed_slots, pos_emb), dim=-1)

        features = self.renderer(mixed_slots)
        recons = self.output_transform(features)
        return {"reconstruction": recons, "masks": attn.transpose(-2, -1)}

class FeatureSimilarity:
    """Compute dot-product based similarity between two sets of features.

    Args:
        normalize: Apply L2 normalization to features before computing dot-product, i.e. compute
            cosine similarity.
        temperature: Divide similarities by this value after computing dot-product.
        threshold: Set values below this threshold to maximum dissimilarity before temperature
            scaling.
        mask_diagonal: Whether to set the diagonal of the similarity matrix to maximum
            dissimilarity after applying temperature scaling.
        softmax: Apply softmax over last dimension after computing similarity.
        sigmoid: Apply sigmoid after computing similarity.
        relative: Whether to transform similarities such that resulting similarity matrix only
            contains similarities spatially around position.
        relative_window_size: Size of relative window.
    """

    def __init__(
        self,
        normalize: bool = True,
        temperature: float = 1.0,
        threshold: Optional[float] = None,
        mask_diagonal: bool = False,
        softmax: bool = False,
    ):
        self.normalize = normalize
        self.temperature = temperature
        self.threshold = threshold
        self.mask_diagonal = mask_diagonal
        self.softmax = softmax

        # Choose padding value such that it indicates maximum dissimilarity
        self.padding_value = -torch.inf if self.softmax else -1.0 / self.temperature

    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        similarity = torch.einsum("bpd, bkd -> bpk", features1, features2)

        if self.threshold is not None:
            similarity[similarity < self.threshold] = self.padding_value

        similarity /= self.temperature

        if self.mask_diagonal:
            diag = torch.diagonal(similarity, dim1=-2, dim2=-1)
            diag[:, :] = self.padding_value

        if self.softmax:
            # if all the values in a row are padding, softmax will return nan.
            # To avoid this, we set the padding values to 0.
            similarity[
                (similarity == self.padding_value)
                .all(dim=-1, keepdim=True)
                .expand(-1, -1, similarity.shape[-1])
            ] = 0.0
            similarity = torch.softmax(similarity, dim=-1)

        return similarity

    def __call__(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            features1 = nn.functional.normalize(features1, p=2.0, dim=-1)
            features2 = nn.functional.normalize(features2, p=2.0, dim=-1)

        return self.compute_similarity(features1, features2)

class FeatureTimeSimilarity(FeatureSimilarity):
    """Compute similaritiy between features over time."""

    def __init__(
        self,
        time_shift: int = 1,
        normalize: bool = True,
        temperature: float = 1.0,
        threshold: Optional[float] = None,
        mask_diagonal: bool = False,
        softmax: bool = False,
    ):
        super().__init__(
            normalize,
            temperature,
            threshold,
            mask_diagonal,
            softmax,
        )
        self.time_shift = time_shift

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        assert features.ndim == 4, "`features` should have shape (batch, frames, positions, dims)"

        if self.normalize:
            features = nn.functional.normalize(features, p=2.0, dim=-1)

        source_features = features[:, : -self.time_shift]
        dest_features = features[:, self.time_shift :]

        source_features = einops.rearrange(source_features, "b t p d -> (b t) p d")
        dest_features = einops.rearrange(dest_features, "b t p d -> (b t) p d")

        similarity = self.compute_similarity(source_features, dest_features)

        similarity = einops.rearrange(similarity, "(b t) p k -> b t p k", b=len(features))

        return similarity

class VIDEOSAUR(nn.Module):
    """
    An implementation of VIDEOSAUR based on the original model
    Encode (f) a video into a set of latent vectors.

    y = f(x), where
        x: (B, T, C, H, W)
        y: (B, T, N, H_out)

    Args:
        input_shape:      (T, C, H, W), the shape of the video
        output_size:      H_out, the latent slots size
    """

    def __init__(
        self,
        resolution,
        encoder_type="vit_base_patch8_224_dino",
        feature_key="vit_block12",
        target_key="vit_block_keys12",
        num_slots=8,
        slot_size=64,
        slot_hidden_size=128,
        pred_layers=1,
        pred_heads=4,
        pred_ffn_dim=512,
        alloc_layers=2,
        alloc_heads=4,
        num_channels=3,
        patch_size=8,
        use_mse= False,
        sim_temp=0.075,
        use_entropy_regularization=False,
        ckpt_path=None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.resolution = resolution
        self.embed_dim = 768 if "vit_base" in encoder_type else 384

        ## ENCODER : DINO model pretrained
        two_layer_mlp = MLP(
            self.embed_dim, # Embed dim
            self.slot_size,
            [self.embed_dim*2],
            True)
        self.encoder = DinoEncoder(
            spatial_flatten=False,
            main_features_key=feature_key,
            model=encoder_type,
            output_transform=two_layer_mlp
        )
        self.target_key = target_key
        self.patch_size = patch_size
        self.num_patches = (self.resolution[0] // patch_size) **2

        ## OBJECT-CENTRIC MODULE : Slot-Attention for Video == SA + Transformer
        self.slot_attn = SlotAttention(
            inp_dim=self.slot_size,
            slot_dim=self.slot_size,
            hidden_dim=slot_hidden_size,
            n_iters=2,
            use_mlp=False
        )
        self.init_slots = RandomInit(self.num_slots, self.slot_size)

        self.predictor = TransformerEncoder(
                self.slot_size,
                pred_layers,
                pred_heads,
        )

        ## DECODER: Spatial-Broadcast Decoder (SBD)
        allocator = TransformerEncoder(
                self.slot_size,
                alloc_layers,
                alloc_heads,
                hidden_dim=pred_ffn_dim,
                memory_dim=self.slot_size,
        )
        renderer = MLP(
            inp_dim=self.slot_size,
            outp_dim=1024,
            hidden_dims=[1024, 1024, 1024],
            final_activation=True
        )

        # Loss fn
        self.use_entropy_regularization = use_entropy_regularization
        print(f"Using entropy regularization: {self.use_entropy_regularization}")
        self.time_similarity = FeatureTimeSimilarity(
            softmax=True,
            temperature=sim_temp,
            threshold=0.0,
        )
        self.ce_loss = nn.CrossEntropyLoss(
            reduction='mean',
        )
        self.use_mse = use_mse
        if self.use_mse:
            self.mse_loss = nn.MSELoss(
                reduction='mean',
            )
            outp_dim = self.num_patches + self.embed_dim
        else:
            outp_dim = self.num_patches
        self.decoder = SlotMixerDecoder(
            inp_dim=self.slot_size,
            outp_dim=outp_dim,
            embed_dim=self.slot_size,
            n_patches=self.num_patches,
            allocator=allocator,
            renderer=renderer,
            renderer_dim=1024,
            use_layer_norms=True,
            pos_embed_mode="add",
        )

        if ckpt_path:
            print(f"Loading model from {ckpt_path}")
            self.load_ckpt(ckpt_path)

    def encode(self, img, prev_slots=None):
        unflatten = False 
        if img.dim() == 5:
            unflatten = True
            B, T = img.shape[:2]
            img = img.flatten(0, 1)
        else:
            B = img.shape[0]
            T = 1
        out_dict = self.encoder(img)
        features = out_dict['features']
        if self.use_mse:
            target = {
                'ce_target': out_dict[self.target_key].detach(),
                'mse_target': out_dict['backbone_features'].detach(),
            }
        else:
            target = out_dict[self.target_key].detach()

        if unflatten:
            features = features.unflatten(0, (B, T)) 
            img = img.unflatten(0, (B, T))
        else:
            features = features.unsqueeze(1)
            img = img.unsqueeze(1)

        all_slots, all_masks = [], []
        if prev_slots is None:
            prev_slots = self.init_slots(B)
        for idx in range(T):
            slots = self.predictor(prev_slots)  # [B, N, C]
            out_dict = self.slot_attn(slots, features[:, idx])
            slots = out_dict['slots']
            masks = out_dict['masks']
            all_slots.append(slots)
            all_masks.append(masks)

            # next timestep
            prev_slots = slots
        slots = torch.stack(all_slots, dim=1).flatten(0, 1)  # [B*T, N, C]
        masks = torch.stack(all_masks, dim=1)
        return slots, masks, target

    def decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        out_dict = self.decoder(slots)
        recons = out_dict['reconstruction']
        masks = out_dict['masks']
        
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
        return recons, masks, slots
    
    def forward(self, img, train=True, prev_slots=None):
        B, T = img.shape[:2]
        h = w = int(self.num_patches**0.5)
        slots, masks_enc, target = self.encode(img, prev_slots=prev_slots)

        out_dict = {
            'slots': slots,
            'feature_map': slots,
            'masks_enc': masks_enc,
            'target': target,
        }
        # Decode
        if train:
            recons, masks_dec, slots = self.decode(slots)
            if isinstance(target, dict):
                target = {k: v.unflatten(0, (B, T)) for k, v in target.items()}
            else:
                target = target.unflatten(0, (B, T))
            attn_mask = masks_enc.flatten(0, 1).unflatten(-1, (h, w)).permute(1, 0, 2, 3)
            assert attn_mask.shape == (self.num_slots, B*T, h, w)
            loss_dict = self.loss_function(recons.unflatten(0, (B, T)), target, attention_mask=attn_mask)
            out_dict['masks_dec'] = masks_dec.unflatten(0, (B, T)).unflatten(-1, (h, w))
            out_dict['recons'] = recons.unflatten(0, (B, T))
            for k, v in loss_dict.items():
                out_dict[k] = v
        return out_dict
    
    def loss_function(self, pred, target, attention_mask=None):
        loss_dict = {}
        # LOSS MSE
        if self.use_mse:
            target_ce = target['ce_target']
            target_mse = target['mse_target']
            mse_slice = slice(0, self.embed_dim)
            ce_slice = slice(self.embed_dim, self.embed_dim+self.num_patches)
            recons_mse = pred[..., mse_slice]
            recons_ce = pred[..., ce_slice]
            loss_mse = self.mse_loss(recons_mse, target_mse)
            loss_dict['mse_loss'] = loss_mse
        else:
            target_ce = target
            recons_ce = pred
        
        # LOSS CE
        pred_ce = recons_ce[: , :-1] # Remove last frame
        pred_ce = einops.rearrange(pred_ce, "b t p d -> b (t p) d") # (batch, positions, dims)
        with torch.no_grad():
            target_ce = self.time_similarity(target_ce)
        target_ce = einops.rearrange(target_ce, "b t p d -> b (t p) d")
        pred_ce = pred_ce.transpose(-2, -1)
        target_ce = target_ce.transpose(-2, -1)
        loss_ce = self.ce_loss(pred_ce, target_ce)
        loss_dict['ce_loss'] = loss_ce
        if self.use_mse:
            loss_dict["loss"] = loss_mse + loss_ce * 0.1 # TODO : add a weight for the loss 
        else:
            loss_dict["loss"] = loss_ce * 0.1

        if self.use_entropy_regularization:
            entropy_loss = torch.mean(torch.square(torch.sum(attention_mask*torch.log(attention_mask + 1e-20), dim=0)))
            loss_dict["entropy_loss"] = entropy_loss
            loss_dict["loss"] += entropy_loss * 0.01
        return loss_dict

    def output_shape(self):
        return self.output_shape
    
    def load_ckpt(self, model_path):
        state_dict = torch.load(model_path, weights_only=False)["state_dict"]
        for key in list(state_dict.keys()):
            new_key = copy.deepcopy(key)
            if "model.encoder" in new_key:
                new_key = new_key.replace("model.encoder", "encoder")
            if "model.decoder" in new_key:
                new_key = new_key.replace("model.decoder", "decoder")
            if "model.predictor" in new_key:
                new_key = new_key.replace("model.predictor", "predictor")
            if "model.slot_attn" in new_key:
                new_key = new_key.replace("model.slot_attn", "slot_attn")
            if "model.init_slots" in new_key:
                new_key = new_key.replace("model.init_slots", "init_slots")
            if ".module" in new_key:
                new_key = new_key.replace(".module", "")
            if "initializer" in new_key:
                new_key = new_key.replace("initializer", "init_slots")
            if "processor.corrector" in new_key:
                new_key = new_key.replace("processor.corrector", "slot_attn")
            if "processor.predictor" in new_key:
                new_key = new_key.replace("processor.predictor", "predictor")
            state_dict[new_key] = state_dict.pop(key)
        self.load_state_dict(state_dict)