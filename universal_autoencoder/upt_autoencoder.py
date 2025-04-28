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

    # coord_dim: int  # Input coordinate dimensions (e.g., 2 for images, 3 for volumes)
    # cond_dim: Optional[int] = None  # Conditioning dimension
    # cond_encoder_features: Sequence[int] = (128, 128)  # Condition encoder sizes
    # nef_num_layers: int = 4
    # nef_hidden_dim: int = 128
    # nef_omega_0: float = 30.0
    # nef_modulation_hidden_dim: int = 128
    # nef_modulation_num_layers: int = 2
    # radius: float = 1.0
    # max_degree: int = 5
    # gnn_dim: int = 128
    # max_supernodes: int = 32
    # enc_dim: int = 128
    # enc_depth: int = 4
    # enc_num_heads: int = 4
    # perc_dim: Optional[int] = None
    # perc_num_heads: Optional[int] = None
    # num_latent_tokens: Optional[int] = None
    # init_weights: str = "truncnormal"
    # # Coordinate transformation parameters
    # output_coord_dim: Optional[int] = None  # Output coordinate dimensions (e.g., 2D)
    # coord_enc_dim: Optional[int] = None  # Dimension for coordinate encoder
    # coord_enc_depth: int = 2  # Depth of coordinate encoder transformer
    # coord_enc_num_heads: int = 4  # Number of heads in coordinate encoder

    def setup(self):

        # UPT encoder
        self.upt_encoder = EncoderSupernodes(
            cfg=self.cfg,
            # input_dim=encoder_supernodes_cfg.coord_dim,
            # ndim=encoder_supernodes_cfg.coord_dim,
            # radius=encoder_supernodes_cfg.radius,
            # max_degree=encoder_supernodes_cfg.max_degree,
            # gnn_dim=encoder_supernodes_cfg.gnn_dim,
            # max_supernodes=encoder_supernodes_cfg.max_supernodes,
            # enc_dim=encoder_supernodes_cfg.enc_dim,
            # enc_depth=encoder_supernodes_cfg.enc_depth,
            # enc_num_heads=encoder_supernodes_cfg.enc_num_heads,
            # perc_dim=encoder_supernodes_cfg.perc_dim,
            # perc_num_heads=encoder_supernodes_cfg.perc_num_heads,
            # num_latent_tokens=encoder_supernodes_cfg.num_latent_tokens,
            # cond_dim=encoder_supernodes_cfg.cond_dim,
            # init_weights=encoder_supernodes_cfg.init_weights,
            # output_coord_dim=encoder_supernodes_cfg.output_coord_dim,
            # coord_enc_dim=encoder_supernodes_cfg.coord_enc_dim,
            # coord_enc_depth=encoder_supernodes_cfg.coord_enc_depth,
            # coord_enc_num_heads=encoder_supernodes_cfg.coord_enc_num_heads,
        )

        self.siren = ModulatedSIREN(
            cfg=self.cfg,
            # output_dim=siren_cfg.output_dim,
            # num_layers=siren_cfg.num_layers,
            # hidden_dim=siren_cfg.hidden_dim,
            # omega_0=siren_cfg.omega_0,
            # modulation_hidden_dim=siren_cfg.modulation_hidden_dim,
            # modulation_num_layers=siren_cfg.modulation_num_layers,
            # shift_modulate=siren_cfg.shift_modulate,
            # scale_modulate=siren_cfg.scale_modulate,
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
    supernode_idxs = jax.random.randint(subkey, (batch_size, num_supernodes), 0, num_points)
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
