import flax.linen as nn
from typing import Sequence, Optional
from absl import app, flags
import jax
import jax.numpy as jnp
from ml_collections import config_flags
from ml_collections import ConfigDict
from universal_autoencoder.upt_encoder import EncoderSupernodesGrid
from universal_autoencoder.siren import ModulatedSIREN


class UniversalAutoencoderGrid(nn.Module):
    """Universal Autoencoder model.

    This model combines a UPT encoder with a SIREN network for neural field generation.
    """

    cfg: ConfigDict

    def setup(self):

        # UPT encoder
        self.upt_encoder = EncoderSupernodesGrid(
            cfg=self.cfg,
        )

        self.siren = ModulatedSIREN(
            cfg=self.cfg,
        )

    def __call__(self, points, supernode_idxs, coords):
        """
        Args:
            points: Shape (batch_size, num_points, coord_dim)
            condition: Optional shape (batch_size, cond_dim)

        Returns:
            out: Output prediction for each point
        """
        # The encoder now returns both conditioning and transformed coordinates
        conditioning = self.upt_encoder(points, supernode_idxs)

        # Use the transformed coordinates with the SIREN network
        out = self.siren(coords, conditioning)

        return out, conditioning


