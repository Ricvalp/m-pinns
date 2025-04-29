import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
from ml_collections import ConfigDict

from universal_autoencoder.upt_decoder import DecoderPerceiver
from universal_autoencoder.transformers import (
    PrenormBlock,
    ContinuousSincosEmbed,
    LinearProjection,
    PerceiverPoolingBlock,
)


def k_nearest_neighbors(x, supernode_idxs, k):
    """JAX implementation of k-nearest neighbors with batch support

    Args:
        x: Shape (batch_size, num_points, ndim)
        supernode_idxs: Shape (batch_size, num_supernodes) - Indices of supernodes
        k: Number of nearest neighbors to consider

    Returns:
        coords: Shape (batch_size, num_supernodes, k, 3) - Coordinates of k nearest neighbors for each supernode
    """

    def process_sample(sample_x, sample_supernodes_idxs):
        # Compute pairwise distances
        dists = jnp.sum((sample_x[:, None, :] - sample_x[None, :, :]) ** 2, axis=-1)
        dists_from_supernodes = dists[sample_supernodes_idxs]

        # Get indices of k nearest neighbors for each supernode
        neighbor_idxs = jnp.argsort(dists_from_supernodes, axis=1)[:, :k]

        return sample_x[neighbor_idxs]

    # Process each sample in the batch
    return jax.vmap(process_sample, in_axes=(0, 0))(x, supernode_idxs)


class SupernodePooling(nn.Module):
    max_degree: int
    input_dim: int
    hidden_dim: int
    ndim: int

    @nn.compact
    def __call__(self, input_points, supernode_idxs):
        """
        Args:
            input_pos: Shape (batch_size, num_points, ndim)
            supernode_idxs: Shape (batch_size, num_supernodes) - Indices of supernodes

        Returns:
            x: Shape (batch_size, max_supernodes, hidden_dim)
        """

        # Radius graph - creates edges between nodes
        supernode_neighbors_points = k_nearest_neighbors(
            x=input_points,
            supernode_idxs=supernode_idxs,
            k=self.max_degree,
        )

        input_proj = LinearProjection(features=self.hidden_dim)(
            supernode_neighbors_points
        )
        pos_embed = ContinuousSincosEmbed(dim=self.hidden_dim, ndim=self.ndim)(
            supernode_neighbors_points
        )

        x = input_proj + pos_embed

        # Message passing network
        message_net = nn.Sequential(
            [
                LinearProjection(features=self.hidden_dim),
                lambda x: nn.gelu(x),
                LinearProjection(features=self.hidden_dim),
            ]
        )

        x = message_net(x)
        x = jnp.mean(x, axis=-2)

        return x


class EncoderSupernodes(nn.Module):
    cfg: ConfigDict

    def setup(self):

        encoder_supernodes_cfg = self.cfg.encoder_supernodes_cfg

        # Supernode pooling with fixed max_supernodes
        self.supernode_pooling = SupernodePooling(
            max_degree=encoder_supernodes_cfg.max_degree,
            input_dim=encoder_supernodes_cfg.input_dim,
            hidden_dim=encoder_supernodes_cfg.gnn_dim,
            ndim=encoder_supernodes_cfg.ndim,
        )

        # Encoder projection
        self.enc_proj = LinearProjection(
            features=encoder_supernodes_cfg.enc_dim,
            use_bias=True,
            optional=True,
        )

        # Transformer blocks for conditioning
        self.enc_blocks = [
            PrenormBlock(
                dim=encoder_supernodes_cfg.enc_dim,
                num_heads=encoder_supernodes_cfg.enc_num_heads,
            )
            for _ in range(encoder_supernodes_cfg.enc_depth)
        ]

        # Perceiver pooling for conditioning
        if encoder_supernodes_cfg.num_latent_tokens is not None:
            self.perceiver = PerceiverPoolingBlock(
                dim=encoder_supernodes_cfg.perc_dim,
                num_heads=encoder_supernodes_cfg.perc_num_heads,
                num_query_tokens=encoder_supernodes_cfg.num_latent_tokens,
                perceiver_kwargs=dict(
                    kv_dim=encoder_supernodes_cfg.enc_dim,
                    init_weights=encoder_supernodes_cfg.init_weights,
                ),
            )
        else:
            self.perceiver = lambda kv: kv

        self.coord_encoder = DecoderPerceiver(
            output_dim=encoder_supernodes_cfg.output_coord_dim,
            ndim=encoder_supernodes_cfg.ndim,
            dim=encoder_supernodes_cfg.perc_dim,
            depth=encoder_supernodes_cfg.perc_depth,
            num_heads=encoder_supernodes_cfg.perc_num_heads,
            perc_dim=encoder_supernodes_cfg.perc_dim,
            perc_num_heads=encoder_supernodes_cfg.perc_num_heads,
        )

        self.latent_token = self.param(
            "latent_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, encoder_supernodes_cfg.enc_dim),
        )

        self.latent_encoder = nn.Sequential(
            [
                PrenormBlock(
                    dim=encoder_supernodes_cfg.enc_dim,
                    num_heads=encoder_supernodes_cfg.enc_num_heads,
                )
                for _ in range(encoder_supernodes_cfg.latent_encoder_depth)
            ]
        )

    def __call__(self, points, supernode_idxs):

        # ----- Conditioning branch -----
        # Supernode pooling
        x = self.supernode_pooling(
            input_points=points,
            supernode_idxs=supernode_idxs,
        )

        # Project to encoder dimension
        x = self.enc_proj(x)

        # Apply transformer blocks
        for block in self.enc_blocks:
            x = block(x)

        tokens = self.perceiver(kv=x)

        coords_2d = self.coord_encoder(tokens, points)

        latent_token = self.latent_token.repeat(tokens.shape[0], axis=0)

        tokens_with_latent = jnp.concatenate([latent_token, tokens], axis=1)

        # Pass through prenorm stack
        tokens_with_latent = self.latent_encoder(tokens_with_latent)

        # Extract latent code from first token
        latent_code = tokens_with_latent[:, 0]

        return coords_2d, latent_code
