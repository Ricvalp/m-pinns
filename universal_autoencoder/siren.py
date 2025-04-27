import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable, Optional, Any, Tuple


class FilmLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.
    
    Applies affine transformation to input features based on conditioning information.
    Each feature dimension is modulated separately with a scale and shift.
    """
    features: int
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x, condition):
        """
        Args:
            x: Input tensor of shape (..., features)
            condition: Conditioning tensor of shape (batch_size, cond_dim)
            
        Returns:
            Modulated tensor of shape (..., features)
        """
        # Project condition to get scales and shifts
        film_params = nn.Dense(features=2 * self.features, use_bias=self.use_bias)(condition)
        
        # Split into scales and shifts
        scales, shifts = jnp.split(film_params, 2, axis=-1)
        
        # Add extra dimensions for broadcasting correctly
        # If x is (batch, ..., features), condition is (batch, cond_dim)
        # Then scales and shifts need to be (batch, 1, ..., features)
        for _ in range(x.ndim - condition.ndim):
            scales = scales[:, None, ...]
            shifts = shifts[:, None, ...]
        
        # Apply modulation: x * scales + shifts
        return x * (scales + 1.0) + shifts


class SirenLayer(nn.Module):
    """SIREN layer: Linear layer followed by sine activation.
    
    As introduced in "Implicit Neural Representations with Periodic Activation Functions"
    https://arxiv.org/abs/2006.09661
    """
    features: int
    w0: float = 30.0
    is_first: bool = False
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x):
        # Initialize first layer differently, as recommended in the SIREN paper
        if self.is_first:
            kernel_init = nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in', distribution='uniform'
            )
        else:
            kernel_init = nn.initializers.variance_scaling(
                scale=1.0 / self.w0, mode='fan_in', distribution='uniform'
            )
        
        # Linear transformation
        x = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=kernel_init
        )(x)
        
        # Sine activation with frequency modulation
        return jnp.sin(self.w0 * x)


class FilmSiren(nn.Module):
    """SIREN MLP with FiLM conditioning.
    
    Sinusoidal neural network with periodic activations that can be modulated
    by conditioning information through FiLM layers.
    """
    features: Sequence[int]  # Hidden layer sizes
    w0: float = 30.0  # Frequency of first layer
    w0_initial: float = 30.0  # Frequency of initial layer
    final_activation: Optional[Callable] = None  # Optional activation for final output
    use_film: bool = True  # Whether to use FiLM conditioning
    
    @nn.compact
    def __call__(self, x, condition=None):
        """
        Args:
            x: Input coordinates tensor of shape (batch_size, ..., input_dim)
            condition: Optional conditioning tensor of shape (batch_size, cond_dim)
            
        Returns:
            Output tensor of shape (batch_size, ..., features[-1])
        """
        if self.use_film and condition is None:
            raise ValueError("FiLM conditioning requires a condition tensor, but none was provided")
            
        # Initial SIREN layer with different initialization
        x = SirenLayer(
            features=self.features[0],
            w0=self.w0_initial,
            is_first=True
        )(x)
        
        # Apply FiLM conditioning to first layer output if requested
        if self.use_film and condition is not None:
            x = FilmLayer(features=self.features[0])(x, condition)
        
        # Hidden layers
        for i, feat in enumerate(self.features[1:]):
            x = SirenLayer(
                features=feat,
                w0=self.w0
            )(x)
            
            # Apply FiLM after each SIREN layer
            if self.use_film and condition is not None:
                x = FilmLayer(features=feat)(x, condition)
        
        # Apply final activation if specified
        if self.final_activation is not None:
            x = self.final_activation(x)
            
        return x


class ConditionEncoder(nn.Module):
    """Encoder for conditioning information.
    
    Maps input condition to a representation suitable for FiLM modulation.
    """
    features: Sequence[int]  # Hidden layer sizes
    
    @nn.compact
    def __call__(self, condition):
        x = condition
        
        # Process through MLP
        for feat in self.features[:-1]:
            x = nn.Dense(features=feat)(x)
            x = nn.relu(x)
            
        # Final layer without activation
        if len(self.features) > 0:
            x = nn.Dense(features=self.features[-1])(x)
            
        return x


class SirenModel(nn.Module):
    """Complete SIREN model with condition encoder.
    
    Combines a condition encoder with a SIREN network for neural field generation.
    """
    coord_dim: int  # Input coordinate dimensions (e.g., 2 for images, 3 for volumes)
    cond_dim: Optional[int] = None  # Conditioning dimension
    cond_encoder_features: Sequence[int] = (128, 128)  # Condition encoder sizes
    siren_features: Sequence[int] = (256, 256, 256, 3)  # SIREN network sizes
    w0: float = 30.0  # Frequency for hidden layers
    w0_initial: float = 30.0  # Frequency for first layer
    
    def setup(self):
        # Setup condition encoder if conditioning is used
        if self.cond_dim is not None:
            self.condition_encoder = ConditionEncoder(features=self.cond_encoder_features)
            use_film = True
        else:
            use_film = False
            
        # Setup SIREN network
        self.siren = FilmSiren(
            features=self.siren_features,
            w0=self.w0,
            w0_initial=self.w0_initial,
            use_film=use_film
        )
    
    def __call__(self, coords, condition=None):
        """
        Args:
            coords: Coordinate tensor of shape (batch_size, ..., coord_dim)
            condition: Optional conditioning tensor of shape (batch_size, cond_dim)
            
        Returns:
            Predicted values at coordinates of shape (batch_size, ..., output_dim)
        """
        # Process condition if provided
        if condition is not None and self.cond_dim is not None:
            encoded_condition = self.condition_encoder(condition)
        else:
            encoded_condition = None
            
        # Generate output using SIREN
        output = self.siren(coords, encoded_condition)
        
        return output


def test_siren_model():
    """Test function for SirenModel"""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create a simple 2D image model with condition
    model = SirenModel(
        coord_dim=2,
        cond_dim=10,
        cond_encoder_features=(64, 128),
        siren_features=(256, 256, 256, 3),  # RGB output
    )
    
    # Create sample inputs
    batch_size = 64
    img_size = 64
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Create normalized pixel coordinates
    y, x = jnp.meshgrid(
        jnp.linspace(-1, 1, img_size),
        jnp.linspace(-1, 1, img_size),
        indexing='ij'
    )
    coords = jnp.stack([x, y], axis=-1)
    coords = jnp.broadcast_to(coords[None], (batch_size, img_size, img_size, 2))
    
    # Create random condition
    condition = jax.random.normal(subkey2, (batch_size, 10))
    
    # Initialize model
    params = model.init(subkey1, coords, condition)
    
    # Forward pass
    output = model.apply(params, coords, condition)
    
    print(f"Input coords shape: {coords.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {(batch_size, img_size, img_size, 3)}")
    
    return output


if __name__ == "__main__":
    test_siren_model()
