from typing import Any, Dict

import jax.numpy as jnp
import flax.linen as nn


class LinearProjection(nn.Module):
    features: int
    use_bias: bool = True
    optional: bool = False

    @nn.compact
    def __call__(self, x):
        if self.optional and x.shape[-1] == self.features:
            return x

        return nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.glorot_normal(),
        )(x)


class ContinuousSincosEmbed(nn.Module):
    dim: int
    ndim: int

    @nn.compact
    def __call__(self, pos):
        batch_size, num_points = pos.shape[0], pos.shape[1]

        # Create sinusoidal position embeddings
        # Each spatial dimension gets dim/ndim features
        emb_dim_per_dim = self.dim // self.ndim
        half_emb_dim = emb_dim_per_dim // 2

        emb = jnp.log(10000.0) / (half_emb_dim - 1)
        emb = jnp.exp(jnp.arange(half_emb_dim) * -emb)

        # Apply to each spatial dimension separately
        embeddings = []
        for i in range(self.ndim):
            # Get position for this dimension
            pos_i = pos[
                ..., i : i + 1
            ]  # Keep dim for broadcasting (batch_size, num_points, 1)

            # Create embeddings for this dimension
            emb_i = pos_i * emb.reshape(
                *([1] * (pos_i.ndim - 1)), half_emb_dim
            )  # (batch_size, num_points, 1, half_emb_dim)
            emb_i = jnp.concatenate(
                [jnp.sin(emb_i), jnp.cos(emb_i)], axis=-1
            )  # (batch_size, num_points, 1, emb_dim_per_dim)

            embeddings.append(emb_i)

        # Concatenate embeddings from all dimensions
        x = jnp.concatenate(
            embeddings, axis=-1
        )  # (batch_size, num_points, emb_dim_per_dim * ndim)

        # If we have leftover dimensions due to integer division, pad with zeros
        if x.shape[-1] < self.dim:
            padding = self.dim - x.shape[-1]
            x = jnp.pad(x, (*([(0, 0)] * (x.ndim - 1)), (0, padding)))

        return x


class PrenormBlock(nn.Module):
    dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # Self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
        )(x, x)
        x = residual + x

        # MLP
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.dim)(x)
        x = residual + x

        return x


class DitBlock(nn.Module):
    dim: int
    num_heads: int
    cond_dim: int
    init_weights: str = "torch"

    @nn.compact
    def __call__(self, x, cond=None):
        # Self-attention with condition
        residual = x
        x = nn.LayerNorm()(x)

        # Condition projection for attention
        cond_attn = nn.Dense(features=self.dim)(cond)
        x = x + cond_attn[:, None, :]

        x = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
        )(x, x)
        x = residual + x

        # MLP with condition
        residual = x
        x = nn.LayerNorm()(x)

        # Condition projection for MLP
        cond_mlp = nn.Dense(features=self.dim)(cond)
        x = x + cond_mlp[:, None, :]

        x = nn.Dense(features=self.dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.dim)(x)
        x = residual + x

        return x


class PerceiverPoolingBlock(nn.Module):
    dim: int
    num_heads: int
    num_query_tokens: int
    perceiver_kwargs: Dict[str, Any] = None

    @nn.compact
    def __call__(self, kv):
        # Initialize learnable query tokens
        query = self.param(
            "query",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_query_tokens, self.dim),
        )

        # Expand query to batch size
        batch_size = kv.shape[0]
        query = jnp.tile(query, (batch_size, 1, 1))

        # Cross-attention
        kv_dim = self.perceiver_kwargs.get("kv_dim", self.dim)
        init_weights = self.perceiver_kwargs.get("init_weights", "torch")

        # Project kv if dimensions don't match
        if kv_dim != self.dim:
            kv = LinearProjection(features=self.dim, init_weights=init_weights)(kv)

        # Apply cross-attention
        x = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
        )(query, kv)

        return x


class DitPerceiverPoolingBlock(nn.Module):
    dim: int
    num_heads: int
    num_query_tokens: int
    perceiver_kwargs: Dict[str, Any] = None

    @nn.compact
    def __call__(self, kv, cond=None):
        # Initialize learnable query tokens
        query = self.param(
            "query",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_query_tokens, self.dim),
        )

        # Expand query to batch size
        batch_size = kv.shape[0]
        query = jnp.tile(query, (batch_size, 1, 1))

        # Condition projection
        cond_dim = self.perceiver_kwargs.get("cond_dim")
        cond_proj = nn.Dense(features=self.dim)(cond)
        query = query + cond_proj[:, None, :]

        # Cross-attention
        kv_dim = self.perceiver_kwargs.get("kv_dim", self.dim)
        init_weights = self.perceiver_kwargs.get("init_weights", "torch")

        # Project kv if dimensions don't match
        if kv_dim != self.dim:
            kv = LinearProjection(features=self.dim, init_weights=init_weights)(kv)

        # Apply cross-attention
        x = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
        )(query, kv)

        return x
