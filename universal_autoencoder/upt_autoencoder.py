import flax.linen as nn
from typing import Sequence, Optional
from absl import app, flags
from ml_collections import config_flags
from ml_collections import ConfigDict
from universal_autoencoder.upt_encoder import EncoderSupernodes
from universal_autoencoder.siren import ModulatedSIREN


class UniversalAutoencoder(nn.Module):
    """Universal Autoencoder model.

    This model combines a UPT encoder with a SIREN network for neural field generation.
    """

    cfg: ConfigDict

    def setup(self):

        # UPT encoder
        self.upt_encoder = EncoderSupernodes(
            cfg=self.cfg,
        )

        self.siren = ModulatedSIREN(
            cfg=self.cfg,
        )

    def __call__(self, points, supernode_idxs):
        """
        Args:
            points: Shape (batch_size, num_points, coord_dim)
            condition: Optional shape (batch_size, cond_dim)

        Returns:
            out: Output prediction for each point
        """
        # The encoder now returns both conditioning and transformed coordinates
        coords, conditioning = self.upt_encoder(points, supernode_idxs)

        # Use the transformed coordinates with the SIREN network
        out = self.siren(coords, conditioning)

        return out


def test_universal_autoencoder(cfg):
    """Test function for UniversalAutoencoder"""
    import jax

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    # Define test parameters
    batch_size = 16
    num_points = 100
    coord_dim = 3  # 3D input coordinates
    num_supernodes = 8

    # Create random input data
    key, subkey = jax.random.split(key)
    points = jax.random.normal(subkey, (batch_size, num_points, coord_dim))

    # Optional condition
    key, subkey = jax.random.split(key)
    supernode_idxs = jax.random.randint(
        subkey, (batch_size, num_supernodes), 0, num_points
    )
    # Initialize model
    model = UniversalAutoencoder(cfg=cfg)

    # Initialize parameters
    key, subkey = jax.random.split(key)
    params = model.init(subkey, points, supernode_idxs)

    # Apply model
    output = model.apply(params, points, supernode_idxs)

    print("\nUniversalAutoencoder initialized and forward pass completed successfully!")
    print(f"Input shape: {points.shape}")
    print(f"Output shape: {output.shape}")
    print(
        f"Expected output shape: (batch_size, num_points, 3) = ({batch_size}, {num_points}, 3)"
    )

    return output


_TASK_FILE = config_flags.DEFINE_config_file(
    "config", default="universal_autoencoder/config.py"
)


def load_cfgs(_TASK_FILE):
    """Load configuration from file."""
    cfg = _TASK_FILE.value
    return cfg


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    test_universal_autoencoder(cfg)


if __name__ == "__main__":
    app.run(main)
