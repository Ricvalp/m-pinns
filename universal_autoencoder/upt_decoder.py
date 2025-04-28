import jax.numpy as jnp
import einops
from flax import linen as nn
from typing import Optional
from universal_autoencoder.upt_encoder import PrenormBlock


class LinearProjection(nn.Module):

    input_dim : int
    output_dim : int
    ndim : Optional[int] = None
    bias : bool = True
    optional : bool = False
    init_weights : str = "xavier_uniform"

    def setup(self):
        if self.optional and self.input_dim == self.output_dim:
            self.proj = nn.Identity()
        elif self.ndim is None:
            self.proj = nn.Dense(self.output_dim, use_bias=self.bias, kernel_init=nn.initializers.xavier_uniform())
        elif self.ndim == 1:
            self.proj = nn.Conv(self.output_dim, kernel_size=1, use_bias=self.bias, kernel_init=nn.initializers.xavier_uniform())
        elif self.ndim == 2:
            self.proj = nn.Conv(self.output_dim, kernel_size=1, use_bias=self.bias, kernel_init=nn.initializers.xavier_uniform())
        elif self.ndim == 3:
            self.proj = nn.Conv(self.output_dim, kernel_size=1, use_bias=self.bias, kernel_init=nn.initializers.xavier_uniform())
        else:
            raise NotImplementedError

    def __call__(self, x):
        return self.proj(x)


class Mlp(nn.Module):
    hidden_dim : int
    out_dim : int
    act_ctor : nn.Module = nn.gelu
    bias : bool = True
    init_weights : str = "xavier_uniform"
    init_last_proj_zero : bool = False

    def setup(self):
        self.fc1 = nn.Dense(self.hidden_dim, bias=self.bias, kernel_init=nn.initializers.normal(stddev=0.02))
        self.fc2 = nn.Dense(self.out_dim, bias=self.bias, kernel_init=nn.initializers.normal(stddev=0.02))
        self.act = self.act_ctor

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LayerScale(nn.Module):
    dim : int
    init_scale : float = 1e-5

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
        self.kv = nn.Dense(self.dim * 2, bias=self.bias, kernel_init=nn.initializers.normal(stddev=0.02))
        self.q = nn.Dense(self.dim, bias=self.bias, kernel_init=nn.initializers.normal(stddev=0.02))
        self.proj = nn.Dense(self.dim, bias=self.bias, kernel_init=nn.initializers.normal(stddev=0.02))

    def __call__(self, q, kv):
        kv = self.kv(kv)
        q = self.q(q)

        # split per head
        q = einops.rearrange(
            q,
            "bs seqlen_q (num_heads head_dim) -> bs num_heads seqlen_q head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k, v = einops.rearrange(
            kv,
            "bs seqlen_kv (two num_heads head_dim) -> two bs num_heads seqlen_kv head_dim",
            two=2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        x = nn.dot_product_attention(q, k, v)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        return x


class PerceiverBlock(nn.Module):
    dim: int
    num_heads: int
    kv_dim: Optional[int] = None
    mlp_hidden_dim: Optional[int] = None
    drop_path: float = 0.
    act_ctor: nn.Module = nn.gelu
    norm_ctor: nn.Module = nn.LayerNorm
    bias: bool = True
    concat_query_to_kv: bool = False
    layerscale: Optional[float] = None
    eps: float = 1e-6
    init_weights: str = "xavier_uniform"
    init_norms: str = "nonaffine"
    init_last_proj_zero: bool = False

    def setup(self):
        mlp_hidden_dim = self.mlp_hidden_dim or self.dim * 4
        self.norm1q = self.norm_ctor(self.dim, eps=self.eps)
        self.norm1kv = self.norm_ctor(self.kv_dim or self.dim, eps=self.eps)
        self.attn = PerceiverAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            bias=self.bias,
        )
        self.ls1 = nn.Identity() if self.layerscale is None else LayerScale(self.dim, init_scale=self.layerscale)

        # self.drop_path1 = DropPath(drop_prob=self.drop_path)
        
        self.norm2 = self.norm_ctor(self.dim, eps=self.eps)
        self.mlp = Mlp(
            hidden_dim=mlp_hidden_dim,
            out_dim=self.dim,
            act_ctor=self.act_ctor,
            init_weights=self.init_weights,
            init_last_proj_zero=self.init_last_proj_zero,
        )
        self.ls2 = nn.Identity() if self.layerscale is None else LayerScale(self.dim, init_scale=self.layerscale)

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
    def __init__(
            self,
            input_dim,
            output_dim,
            ndim,
            dim,
            depth,
            num_heads,
            unbatch_mode="dense_to_sparse_unpadded",
            perc_dim=None,
            perc_num_heads=None,
            cond_dim=None,
            init_weights="truncnormal002",
            **kwargs,
    ):
        super().__init__(**kwargs)
        perc_dim = perc_dim or dim
        perc_num_heads = perc_num_heads or num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights
        self.unbatch_mode = unbatch_mode

        # input projection
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)

        # blocks
        if cond_dim is None:
            block_ctor = VitBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(
                    dim=dim,
                    num_heads=num_heads,
                    init_weights=init_weights,
                )
                for _ in range(depth)
            ],
        )

        # prepare perceiver
        self.pos_embed = ContinuousSincosEmbed(
            dim=perc_dim,
            ndim=ndim,
        )
        if cond_dim is None:
            block_ctor = PerceiverBlock
        else:
            block_ctor = partial(DitPerceiverBlock, cond_dim=cond_dim)

        # decoder
        self.query_proj = nn.Sequential(
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
        )
        self.perc = block_ctor(dim=perc_dim, kv_dim=dim, num_heads=perc_num_heads, init_weights=init_weights)
        self.pred = nn.Sequential(
            nn.LayerNorm(perc_dim, eps=1e-6),
            LinearProjection(perc_dim, output_dim, init_weights=init_weights),
        )

    def forward(self, x, output_pos, condition=None):
        # check inputs
        assert x.ndim == 3, "expected shape (batch_size, num_latent_tokens, dim)"
        assert output_pos.ndim == 3, "expected shape (batch_size, num_outputs, dim) num_outputs might be padded"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # input projection
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        # create query
        query = self.pos_embed(output_pos)
        query = self.query_proj(query)

        x = self.perc(q=query, kv=x, **cond_kwargs)
        x = self.pred(x)
        if self.unbatch_mode == "dense_to_sparse_unpadded":
            # dense to sparse where no padding needs to be considered
            x = einops.rearrange(
                x,
                "batch_size seqlen dim -> (batch_size seqlen) dim",
            )
        elif self.unbatch_mode == "image":
            # rearrange to square image
            height = math.sqrt(x.size(1))
            assert height.is_integer()
            x = einops.rearrange(
                x,
                "batch_size (height width) dim -> batch_size dim height width",
                height=int(height),
            )
        else:
            raise NotImplementedError(f"invalid unbatch_mode '{self.unbatch_mode}'")

        return x
    

class DecoderPerceiver(nn.Module):
    output_dim : int
    ndim : int
    dim : int
    depth : int
    num_heads : int
    unbatch_mode : str
    perc_dim : int
    perc_num_heads : int
    cond_dim : int
    init_weights : str

    def setup(self):
        perc_dim = self.perc_dim or self.dim
        perc_num_heads = self.perc_num_heads or self.num_heads

        
        self.blocks = nn.Sequential(
            *[
                PrenormBlock(dim=self.dim, num_heads=self.num_heads)
                for _ in range(self.depth)
            ],
        )

        # prepare perceiver
        self.pos_embed = ContinuousSincosEmbed(
            dim=self.perc_dim,
            ndim=self.ndim,
        )
        if self.cond_dim is None:
            block_ctor = PerceiverBlock
        else:
            block_ctor = partial(DitPerceiverBlock, cond_dim=self.cond_dim)

        # decoder
        self.query_proj = nn.Sequential(
            LinearProjection(self.perc_dim, self.perc_dim, init_weights=self.init_weights),
            nn.gelu,
            LinearProjection(self.perc_dim, self.perc_dim, init_weights=self.init_weights),
        )
        self.perc = block_ctor(dim=self.perc_dim, kv_dim=self.dim, num_heads=self.perc_num_heads, init_weights=self.init_weights)
        self.pred = nn.Sequential(
            nn.LayerNorm(epsilon=1e-6),
            LinearProjection(self.perc_dim, self.output_dim, init_weights=self.init_weights),
        )

    def __call__(self, x, output_pos, condition=None):

        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # input projection
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        # create query
        query = self.pos_embed(output_pos)
        query = self.query_proj(query)

        x = self.perc(q=query, kv=x, **cond_kwargs)
        x = self.pred(x)
        if self.unbatch_mode == "dense_to_sparse_unpadded":
            # dense to sparse where no padding needs to be considered
            x = einops.rearrange(
                x,
                "batch_size seqlen dim -> (batch_size seqlen) dim",
            )
        elif self.unbatch_mode == "image":
            # rearrange to square image
            height = math.sqrt(x.size(1))
            assert height.is_integer()
            x = einops.rearrange(
                x,
                "batch_size (height width) dim -> batch_size dim height width",
                height=int(height),
            )
        else:
            raise NotImplementedError(f"invalid unbatch_mode '{self.unbatch_mode}'")

        return x
    
