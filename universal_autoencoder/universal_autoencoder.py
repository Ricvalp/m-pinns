import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable, Optional, Any, Tuple

from upt_encoder import BatchEncoderSupernodes
from siren import SirenModel


class UniversalAutoencoder(nn.Module):
    """Universal Autoencoder model.
    
    This model combines a UPT encoder with a SIREN network for neural field generation.
    """
    coord_dim: int  # Input coordinate dimensions (e.g., 2 for images, 3 for volumes)
    cond_dim: Optional[int] = None  # Conditioning dimension
    cond_encoder_features: Sequence[int] = (128, 128)  # Condition encoder sizes
    siren_features: Sequence[int] = (256, 256, 256, 3)  # SIREN network sizes
    w0: float = 30.0  # Frequency for hidden layers
    w0_initial: float = 30.0  # Frequency for initial layer
    radius: float = 1.0
    max_degree: int = 5
    gnn_dim: int = 128
    enc_dim: int = 128
    enc_depth: int = 4
    enc_num_heads: int = 4
    perc_dim: Optional[int] = None
    perc_num_heads: Optional[int] = None
    num_latent_tokens: Optional[int] = None
    init_weights: str = "truncnormal"
    # Coordinate transformation parameters
    output_coord_dim: Optional[int] = None  # Output coordinate dimensions (e.g., 2D)
    coord_enc_dim: Optional[int] = None  # Dimension for coordinate encoder
    coord_enc_depth: int = 2  # Depth of coordinate encoder transformer
    coord_enc_num_heads: int = 4  # Number of heads in coordinate encoder

    def setup(self):
        # UPT encoder
        self.upt_encoder = BatchEncoderSupernodes(
            input_dim=self.coord_dim,
            ndim=self.coord_dim,
            radius=self.radius,
            max_degree=self.max_degree,
            gnn_dim=self.gnn_dim,
            enc_dim=self.enc_dim,
            enc_depth=self.enc_depth,
            enc_num_heads=self.enc_num_heads,
            perc_dim=self.perc_dim,
            perc_num_heads=self.perc_num_heads,
            num_latent_tokens=self.num_latent_tokens,
            cond_dim=self.cond_dim,
            init_weights=self.init_weights,
            output_coord_dim=self.output_coord_dim,
            coord_enc_dim=self.coord_enc_dim,
            coord_enc_depth=self.coord_enc_depth,
            coord_enc_num_heads=self.coord_enc_num_heads
        )

        # SIREN network - adjust coordinate dimension based on output_coord_dim
        siren_coord_dim = self.output_coord_dim if self.output_coord_dim is not None else self.coord_dim
        self.siren = SirenModel(
            coord_dim=siren_coord_dim,
            cond_dim=self.cond_dim,
            cond_encoder_features=self.cond_encoder_features,
            siren_features=self.siren_features,
            w0=self.w0,
            w0_initial=self.w0_initial
        )

    def __call__(self, points, condition=None):
        """
        Args:
            points: Shape (batch_size, num_points, coord_dim)
            condition: Optional shape (batch_size, cond_dim)
            
        Returns:
            out: Output prediction for each point
        """
        # The encoder now returns both conditioning and transformed coordinates
        conditioning, coords = self.upt_encoder(points, condition)
        
        # Use the transformed coordinates with the SIREN network
        out = self.siren(coords, conditioning)

        return out


def test_universal_autoencoder():
    """Test function for UniversalAutoencoder with coordinate transformation"""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define test parameters
    batch_size = 4
    num_points = 1000  # Smaller number for testing
    input_dim = 3
    output_dim = 3  # RGB output
    
    # Create 3D input points and a condition
    key, subkey1, subkey2 = jax.random.split(key, 3)
    input_points = jax.random.normal(subkey1, (batch_size, num_points, input_dim))
    condition = jax.random.normal(subkey2, (batch_size, 16))
    
    print(f"Input points shape: {input_points.shape}")
    print(f"Condition shape: {condition.shape}")
    
    # Create model with coordinate transformation (3D to 2D)
    model = UniversalAutoencoder(
        coord_dim=input_dim,
        cond_dim=16,
        cond_encoder_features=(64, 128),
        siren_features=(256, 256, 256, output_dim),
        radius=0.5,
        max_degree=5,
        gnn_dim=64,
        enc_dim=128,
        enc_depth=2,
        enc_num_heads=4,
        perc_dim=128,
        perc_num_heads=4,
        num_latent_tokens=8,
        output_coord_dim=2,  # Transform from 3D to 2D
        coord_enc_dim=64,
        coord_enc_depth=2,
        coord_enc_num_heads=4
    )
    
    try:
        # Initialize parameters
        key, subkey = jax.random.split(key)
        params = model.init(subkey, input_points, condition)
        
        # Apply model
        output = model.apply(params, input_points, condition)
        
        print("\nModel initialized and forward pass completed successfully!")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: (batch_size, num_points, output_dim) = ({batch_size}, {num_points}, {output_dim})")
        
        # Test without coordinate transformation (using original 3D coordinates)
        model_no_transform = UniversalAutoencoder(
            coord_dim=input_dim,
            cond_dim=16,
            cond_encoder_features=(64, 128),
            siren_features=(256, 256, 256, output_dim),
            radius=0.5,
            max_degree=5,
            gnn_dim=64,
            enc_dim=128,
            enc_depth=2,
            enc_num_heads=4,
            perc_dim=128,
            perc_num_heads=4,
            num_latent_tokens=8,
            # No output_coord_dim specified
        )
        
        key, subkey = jax.random.split(key)
        params_no_transform = model_no_transform.init(subkey, input_points, condition)
        output_no_transform = model_no_transform.apply(params_no_transform, input_points, condition)
        
        print("\nModel without coordinate transformation tested successfully!")
        print(f"Output shape: {output_no_transform.shape}")
        print(f"Expected output shape: (batch_size, num_points, output_dim) = ({batch_size}, {num_points}, {output_dim})")
        
        return output, output_no_transform
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    test_universal_autoencoder()
