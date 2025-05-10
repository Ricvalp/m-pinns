import jax.numpy as jnp
import einops
from flax import linen as nn
from typing import Optional
from universal_autoencoder.transformers import (
    PrenormBlock,
    ContinuousSincosEmbed,
    LinearProjection,
)


class Mlp(nn.Module):
    hidden_dim: int
    out_dim: int
    act_ctor: nn.Module = nn.gelu
    bias: bool = True

    def setup(self):
        self.fc1 = nn.Dense(
            self.hidden_dim,
            use_bias=self.bias,
            kernel_init=nn.initializers.glorot_normal(),
        )
        self.fc2 = nn.Dense(
            self.out_dim,
            use_bias=self.bias,
            kernel_init=nn.initializers.glorot_normal(),
        )
        self.act = self.act_ctor

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LayerScale(nn.Module):
    dim: int
    init_scale: float = 1e-5

    def setup(self):
        if self.init_scale is None:
            self.gamma = None
        else:
            self.gamma = nn.Parameter(jnp.full(self.dim, self.init_scale))

    def __call__(self, x):
        if self.gamma is None:
            return x
        return x * self.gamma


class PerceiverAttention(nn.Module):
    dim: int
    num_heads: int = 8
    bias: bool = True

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.kv = nn.Dense(
            self.dim * 2,
            use_bias=self.bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )
        self.q = nn.Dense(
            self.dim,
            use_bias=self.bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )
        self.proj = nn.Dense(
            self.dim,
            use_bias=self.bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )

    def __call__(self, q, kv):
        kv = self.kv(kv)
        q = self.q(q)

        # split per head
        q = einops.rearrange(
            q,
            "bs seqlen_q (num_heads head_dim) -> bs seqlen_q num_heads head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k, v = einops.rearrange(
            kv,
            "bs seqlen_kv (two num_heads head_dim) -> two bs seqlen_kv num_heads head_dim",
            two=2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        x = nn.dot_product_attention(q, k, v)
        x = einops.rearrange(
            x, "bs seqlen num_heads head_dim -> bs seqlen (num_heads head_dim)"
        )
        x = self.proj(x)
        return x


class PerceiverBlock(nn.Module):
    dim: int
    num_heads: int
    kv_dim: Optional[int] = None
    mlp_hidden_dim: Optional[int] = None
    drop_path: float = 0.0
    act_ctor: nn.Module = nn.gelu
    norm_ctor: nn.Module = nn.LayerNorm
    bias: bool = True
    concat_query_to_kv: bool = False
    layerscale: Optional[float] = None
    eps: float = 1e-6

    def setup(self):
        mlp_hidden_dim = self.mlp_hidden_dim or self.dim * 4
        self.norm1q = self.norm_ctor(epsilon=self.eps)
        self.norm1kv = self.norm_ctor(epsilon=self.eps)
        self.attn = PerceiverAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            bias=self.bias,
        )
        self.ls1 = lambda x: (
            x
            if self.layerscale is None
            else LayerScale(self.dim, init_scale=self.layerscale)
        )

        # self.drop_path1 = DropPath(drop_prob=self.drop_path)

        self.norm2 = self.norm_ctor(epsilon=self.eps)
        self.mlp = Mlp(
            hidden_dim=mlp_hidden_dim,
            out_dim=self.dim,
            act_ctor=self.act_ctor,
        )
        self.ls2 = lambda x: (
            x
            if self.layerscale is None
            else LayerScale(self.dim, init_scale=self.layerscale)
        )

        # self.drop_path2 = DropPath(drop_prob=self.drop_path)

    def _attn_residual_path(self, q, kv):
        return self.ls1(self.attn(q=self.norm1q(q), kv=self.norm1kv(kv)))

    def _mlp_residual_path(self, x):
        return self.ls2(self.mlp(self.norm2(x)))

    def __call__(self, q, kv):
        q = self._attn_residual_path(q, kv)
        q = self._mlp_residual_path(q)
        return q


class DecoderPerceiver(nn.Module):
    output_dim: int
    ndim: int
    dim: int
    depth: int
    num_heads: int
    perc_dim: int
    perc_num_heads: int

    def setup(self):

        self.input_proj = LinearProjection(self.dim)

        self.blocks = nn.Sequential(
            [
                PrenormBlock(dim=self.dim, num_heads=self.num_heads)
                for _ in range(self.depth)
            ],
        )

        # prepare perceiver
        self.pos_embed = ContinuousSincosEmbed(
            dim=self.perc_dim,
            ndim=self.ndim,
        )

        # decoder
        self.query_proj = nn.Sequential(
            [
                LinearProjection(self.perc_dim),
                nn.gelu,
                LinearProjection(self.perc_dim),
            ]
        )
        self.perc = PerceiverBlock(
            dim=self.perc_dim,
            kv_dim=self.dim,
            num_heads=self.perc_num_heads,
        )
        self.pred = LinearProjection(self.output_dim)

    def __call__(self, x, output_coords):

        # input projection
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x)

        # create query
        query = self.pos_embed(output_coords)
        query = self.query_proj(query)

        x = self.perc(q=query, kv=x)
        x = self.pred(x)

        x = (nn.sigmoid(x) * 2.0) - 1.0

        return x
