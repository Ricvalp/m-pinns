import math
from typing import Any, Callable, Dict, Sequence, Union

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import dtypes, random
from jax._src.typing import Array
from jax.nn.initializers import Initializer

KeyArray = Array


class MLP(nn.Module):
    """Simple MLP with custom activation function."""

    layer_sizes: Sequence[int]
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for i, size in enumerate(self.layer_sizes[:-1]):
            x = nn.Dense(
                size,
                kernel_init=jax.nn.initializers.truncated_normal(
                    1 / jnp.sqrt(x.shape[-1])
                ),
                bias_init=jax.nn.initializers.zeros,
                name=f"mlp_linear_{i}",
            )(x)
            x = self.activation(x)
        return nn.Dense(
            self.layer_sizes[-1],
            kernel_init=jax.nn.initializers.truncated_normal(1 / jnp.sqrt(x.shape[-1])),
            bias_init=jax.nn.initializers.zeros,
            name=f"mlp_linear_{len(self.layer_sizes) - 1}",
        )(x)


class PosEmb(nn.Module):
    embedding_dim: int
    freq: float

    @nn.compact
    def __call__(self, coords):
        emb = nn.Dense(
            self.embedding_dim // 2,
            kernel_init=nn.initializers.normal(self.freq),
            use_bias=False,
        )(jnp.pi * (coords + 1))
        return nn.Dense(self.embedding_dim)(
            jnp.sin(jnp.concatenate([coords, emb, emb + jnp.pi / 2.0], axis=-1))
        )


def _compute_fans(
    shape,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Union[int, Sequence[int]] = (),
):
    """Compute effective input and output sizes for a linear or convolutional layer.

    Axes not in in_axis, out_axis, or batch_axis are assumed to constitute the "receptive field" of
    a convolution (kernel spatial dimensions).
    """
    if len(shape) <= 1:
        raise ValueError(
            f"Can't compute input and output sizes of a {shape.rank}"
            "-dimensional weights tensor. Must be at least 2D."
        )

    if isinstance(in_axis, int):
        in_size = shape[in_axis]
    else:
        in_size = math.prod([shape[i] for i in in_axis])
    if isinstance(out_axis, int):
        out_size = shape[out_axis]
    else:
        out_size = math.prod([shape[i] for i in out_axis])
    if isinstance(batch_axis, int):
        batch_size = shape[batch_axis]
    else:
        batch_size = math.prod([shape[i] for i in batch_axis])
    receptive_field_size = np.prod(shape) / in_size / out_size / batch_size
    fan_in = in_size * receptive_field_size
    fan_out = out_size * receptive_field_size
    return fan_in, fan_out


def custom_uniform(
    numerator: Any = 6,
    mode="fan_in",
    dtype: Any = jnp.float_,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    distribution="uniform",
) -> Initializer:
    """Builds an initializer that returns real uniformly-distributed random arrays.

    Args:
      scale: the upper and lower bound of the random distribution.
      dtype: optional; the initializer's default dtype.

    Returns:
      An initializer that returns arrays whose values are uniformly distributed in
      the range ``[-range, range)``.
    """

    def init(key: KeyArray, shape: Array, dtype: Any = dtype) -> Any:
        dtype = dtypes.canonicalize_dtype(dtype)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis, batch_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")
        if distribution == "uniform":
            return random.uniform(
                key,
                shape,
                dtype,
                minval=-jnp.sqrt(numerator / denominator),
                maxval=jnp.sqrt(numerator / denominator),
            )
        elif distribution == "normal":
            return random.normal(key, shape, dtype) * jnp.sqrt(numerator / denominator)
        elif distribution == "uniform_squared":
            return random.uniform(
                key,
                shape,
                dtype,
                minval=-numerator / denominator,
                maxval=numerator / denominator,
            )
        else:
            raise ValueError(
                f"invalid distribution for variance scaling initializer: {distribution}"
            )

    return init


class LinearModulation(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:
        # Compute the modulation weights
        mod_weights = nn.Dense(2 * x.shape[-1])(latent)
        # split into bias and scale
        bias, scale = jnp.split(mod_weights, 2, axis=-1)
        # Apply the modulation
        x = x * (scale + 1) + bias
        x = nn.Dense(self.output_dim)(x)
        return x


class SirenLayer(nn.Module):
    output_dim: int
    omega_0: float
    is_first_layer: bool = False
    is_last_layer: bool = False
    apply_activation: bool = True

    def setup(self):
        c = 1 if self.is_first_layer else 6 / self.omega_0**2
        distrib = "uniform_squared" if self.is_first_layer else "uniform"
        self.linear = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=custom_uniform(
                numerator=c, mode="fan_in", distribution=distrib
            ),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        after_linear = self.linear(x)
        if self.apply_activation:
            return jnp.sin(self.omega_0 * after_linear)
        else:
            return after_linear


class LatentToModulation(nn.Module):
    layer_sizes: Sequence[int]
    num_modulation_layers: int
    modulation_dim: int
    input_dim: int
    shift_modulate: bool = True
    scale_modulate: bool = True
    activation: Callable[[Array], Array] = nn.relu

    def setup(self):
        if self.shift_modulate and self.scale_modulate:
            self.modulations_per_unit = 2
        elif self.shift_modulate or self.scale_modulate:
            self.modulations_per_unit = 1
        else:
            raise ValueError(
                "At least one of shift_modulate or scale_modulate must be True"
            )

        self.modulations_per_layer = self.modulations_per_unit * self.modulation_dim
        self.modulation_output_size = (
            self.modulations_per_layer * self.num_modulation_layers
        )

        # create a MLP to process the latent vector based on self.layer_sizes and self.modulation_output_size
        self.mlp = MLP(
            layer_sizes=self.layer_sizes + (self.modulation_output_size,),
            activation=self.activation,
        )

    def __call__(self, x: Array) -> Dict[str, Array]:
        x = self.mlp(x)
        # Split the output into scale and shift modulations
        if self.modulations_per_unit == 2:
            scale, shift = jnp.split(x, 2, axis=-1)
            scale = (
                scale.reshape(
                    (
                        *x.shape[:-1],
                        self.num_modulation_layers,
                        self.modulation_dim,
                    )
                )
                + 1
            )
            shift = shift.reshape(
                (
                    *x.shape[:-1],
                    self.num_modulation_layers,
                    self.modulation_dim,
                )
            )
            return {"scale": scale, "shift": shift}
        else:
            x = x.reshape(
                (
                    *x.shape[:-1],
                    self.num_modulation_layers,
                    self.modulation_dim,
                )
            )
            if self.shift_modulate:
                return {"shift": x}
            elif self.scale_modulate:
                return {"scale": x + 1}


class ModulatedSIREN(nn.Module):
    cfg: ConfigDict

    def setup(self):
        modulated_siren_cfg = self.cfg.modulated_siren_cfg
        self.num_layers = modulated_siren_cfg.num_layers
        self.hidden_dim = modulated_siren_cfg.hidden_dim
        self.omega_0 = modulated_siren_cfg.omega_0
        self.output_dim = modulated_siren_cfg.output_dim

        self.modulator = LatentToModulation(
            input_dim=1,
            layer_sizes=[modulated_siren_cfg.hidden_dim]
            * modulated_siren_cfg.modulation_num_layers,
            num_modulation_layers=modulated_siren_cfg.num_layers - 1,
            modulation_dim=modulated_siren_cfg.hidden_dim,
            scale_modulate=modulated_siren_cfg.scale_modulate,
            shift_modulate=modulated_siren_cfg.shift_modulate,
        )

        self.kernel_net = (
            [
                SirenLayer(
                    output_dim=modulated_siren_cfg.hidden_dim,
                    omega_0=modulated_siren_cfg.omega_0,
                    is_first_layer=True,
                    apply_activation=False,
                )
            ]
            + [
                SirenLayer(
                    output_dim=modulated_siren_cfg.hidden_dim,
                    omega_0=modulated_siren_cfg.omega_0,
                    is_first_layer=False,
                    apply_activation=False,
                )
                for _ in range(self.num_layers - 2)
            ]
            + [
                nn.Dense(
                    features=modulated_siren_cfg.output_dim,
                    use_bias=True,
                    kernel_init=custom_uniform(
                        numerator=6 / modulated_siren_cfg.omega_0**2,
                        mode="fan_in",
                        distribution="uniform",
                    ),
                    bias_init=nn.initializers.zeros,
                )
            ]
        )

    def __call__(self, x, latent):
        modulations = self.modulator(latent)

        for layer_num, layer in enumerate(self.kernel_net):
            x = layer(x)
            if layer_num < self.num_layers - 1:
                x = self.modulate(x, modulations, layer_num)
                x = jnp.sin(self.omega_0 * x)

        return x

    def modulate(
        self, x: Array, modulations: Dict[str, Array], layer_num: int
    ) -> Array:
        """Modulates input according to modulations.

        Args:
            x: Hidden features of MLP.
            modulations: Dict with keys 'scale' and 'shift' (or only one of them)
            containing modulations.

        Returns:
            Modulated vector.
        """
        if "scale" in modulations:
            x = modulations["scale"][..., layer_num, :][..., None, :] * x
        if "shift" in modulations:
            x = x + modulations["shift"][..., layer_num, :][..., None, :]
        return x
