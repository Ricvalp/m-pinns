import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Any
from jax._src import core
from jax._src import dtypes
from jax._src.typing import Array


def sigma(x):
    return nn.sigmoid(x)


def sigma2(x, alpha=0.1):
    return nn.gelu(x)


class RFFEmbedding(nn.Module):
    input_dim: int
    output_dim: int
    sigma: float = 1.0

    def setup(self):
        self.W = self.param(
            "W",
            lambda key, shape: self.sigma * jax.random.normal(key, shape),
            (self.output_dim // 2, self.input_dim),
        )
        self.b = self.param(
            "b",
            lambda key, shape: 2 * jnp.pi * jax.random.uniform(key, shape),
            (self.output_dim // 2,),
        )

    @nn.compact
    def __call__(self, x):
        projection = jnp.dot(x, self.W.T) + self.b
        concat_features = jnp.concatenate(
            [jnp.cos(projection), jnp.sin(projection)], axis=-1
        )
        return concat_features


class Encoder(nn.Module):

    n_hidden: int
    n_latent: int
    rff_dim: int = 128  # dimension for RFF embedding
    init_scale: float = 1.5
    def setup(self):
        self.rff = RFFEmbedding(3, self.rff_dim)
        self.dense1 = nn.Dense(self.n_hidden)
        self.dense2 = nn.Dense(self.n_hidden)
        self.dense3 = nn.Dense(self.n_latent)
        self.scale = self.param("scale", lambda key, shape: jnp.ones(shape) * self.init_scale, ())

    def __call__(self, x):
        x = self.rff(x)
        x = self.dense1(x)
        x = sigma2(x)
        x = self.dense2(x)
        x = sigma2(x)
        x = self.dense3(x)
        x = sigma(x)
        x = x * self.scale
        return x


class Decoder(nn.Module):

    n_hidden: int
    n_out: int = 3  # dimension of the ambient space
    rff_dim: int = 128  # dimension for RFF embedding

    def setup(self):
        self.rff = RFFEmbedding(2, self.rff_dim)
        self.dense1 = nn.Dense(self.n_hidden)
        self.dense2 = nn.Dense(self.n_hidden)
        self.dense3 = nn.Dense(self.n_out)

    def __call__(self, x):
        x = self.rff(x)
        x = self.dense1(x)
        x = sigma2(x)
        x = self.dense2(x)
        x = sigma2(x)
        x = self.dense3(x)
        return x


class AutoEncoder(nn.Module):

    n_hidden: int
    rff_dim: int = 128
    init_scale: float = 1.5
    def setup(self):
        self.encoder = Encoder(self.n_hidden, 2, self.rff_dim, self.init_scale, name="E")
        self.decoder = Decoder(self.n_hidden, 3, self.rff_dim, name="D")

    def __call__(self, x):

        z = self.encoder(x)
        out = self.decoder(z)

        return out, z


class AutoDecoder(nn.Module):
    init_points: jnp.ndarray
    n_hidden: int
    rff_dim: int = 128

    def setup(self):
        self.decoder = Decoder(
            n_hidden=self.n_hidden, rff_dim=self.rff_dim, n_out=3, name="D"
        )
        init = points_init(self.init_points)
        self.points = self.param("points", init, self.init_points.shape)

    def __call__(self):
        return self.decoder(self.points)


def points_init(points):
    def init(key, shape, dtype=jnp.float32):
        return points

    return init


if __name__ == "__main__":
    points = jax.random.normal(jax.random.PRNGKey(0), (6, 100, 2))

    model = AutoDecoder(points, 128, 3)
    params = model.init(jax.random.PRNGKey(0))

    out = model.apply(params)
    print(out.shape)
