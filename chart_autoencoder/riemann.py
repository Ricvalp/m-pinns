import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import vmap
from flax.training import checkpoints
import pickle
from pathlib import Path
import optax
from flax import linen as nn
from flax.training import train_state

from chart_autoencoder.models import Decoder, Encoder
from chart_autoencoder.utils import ModelCheckpoint

from universal_autoencoder import UniversalAutoencoder, ModulatedSIREN


class InducedRiemannianMetric(nn.Module):
    phi: None

    def setup(self):
        self.jac = jax.vmap(jax.jacfwd(self.phi), (0))

    def __call__(self, z):
        return self.jac(z).transpose(0, 2, 1) @ self.jac(z)


class InducedInverseRiemannianMetric(nn.Module):
    phi: None

    def setup(self):
        self.jac = jax.vmap(jax.jacfwd(self.phi), (0))

    def __call__(self, z):
        return jnp.linalg.inv(self.jac(z).transpose(0, 2, 1) @ self.jac(z))


def sqrt_det_g(g):
    return jax.jit(lambda x: jnp.sqrt(jnp.linalg.det(g(x))))


def get_sqrt_det_g(phi):
    g_phi = InducedRiemannianMetric(phi=phi)
    g_phi_fn = lambda z: g_phi.apply({}, z)
    return sqrt_det_g(g=g_phi_fn)


def get_metric(phi, inverse=False):
    if inverse:
        g_phi = InducedInverseRiemannianMetric(phi=phi)
    else:
        g_phi = InducedRiemannianMetric(phi=phi)
    g_phi_fn = lambda z: g_phi.apply({}, z)
    return jax.jit(g_phi_fn)


def get_metric_tensor_and_sqrt_det_g(cfg, step, inverse=False):

    checkpointer = ModelCheckpoint(
        Path(cfg.checkpoint.checkpoint_path).absolute(), overwrite=False
    )
    params = checkpointer.load_checkpoint(step=step)

    e_params = params["E"]
    d_params = params["D"]

    n_hidden = cfg.model.n_hidden

    encoder = Encoder(
        n_hidden=n_hidden,
        n_latent=2,
    )
    decoder = Decoder(
        n_hidden=n_hidden,
        n_out=3,
    )

    def induced_riemannian_metric(params, z):
        phi = lambda p, x: decoder.apply({"params": p}, x)
        jac = jax.vmap(jax.jacfwd(phi, argnums=1), (None, 0))
        return jac(params, z).transpose(0, 2, 1) @ jac(params, z)

    def induced_inverse_riemannian_metric(params, z):
        phi = lambda p, x: decoder.apply({"params": p}, x)
        jac = jax.vmap(jax.jacfwd(phi, argnums=1), (None, 0))
        return jnp.linalg.inv(jac(params, z).transpose(0, 2, 1) @ jac(params, z))

    def sqrt_det_g(params, z):
        return jnp.sqrt(jnp.linalg.det(induced_riemannian_metric(params, z)))

    if inverse:
        return (
            jax.jit(induced_inverse_riemannian_metric),
            jax.jit(sqrt_det_g),
            encoder,
            decoder,
        ), (e_params, d_params)
    else:
        return (
            jax.jit(induced_riemannian_metric),
            jax.jit(sqrt_det_g),
            encoder,
            decoder,
        ), (e_params, d_params)


def get_metric_tensor_and_sqrt_det_g_autodecoder(cfg, step, inverse=False):

    checkpointer = ModelCheckpoint(
        Path(cfg.checkpoint.checkpoint_path).absolute(), overwrite=False
    )
    d_params = checkpointer.load_checkpoint(step=step)

    decoder = Decoder(
        n_hidden=cfg.model.n_hidden,
        rff_dim=cfg.model.rff_dim,
        n_out=3,
    )

    def induced_riemannian_metric(params, z):
        phi = lambda p, x: decoder.apply({"params": p}, x)
        jac = jax.vmap(jax.jacfwd(phi, argnums=1), (None, 0))
        return jac(params, z).transpose(0, 2, 1) @ jac(params, z)

    def induced_inverse_riemannian_metric(params, z):
        phi = lambda p, x: decoder.apply({"params": p}, x)
        jac = jax.vmap(jax.jacfwd(phi, argnums=1), (None, 0))
        return jnp.linalg.inv(jac(params, z).transpose(0, 2, 1) @ jac(params, z))

    def sqrt_det_g(params, z):
        return jnp.sqrt(jnp.linalg.det(induced_riemannian_metric(params, z)))

    if inverse:
        return (
            jax.jit(induced_inverse_riemannian_metric),
            jax.jit(sqrt_det_g),
            decoder,
        ), d_params
    else:
        return (
            jax.jit(induced_riemannian_metric),
            jax.jit(sqrt_det_g),
            decoder,
        ), d_params





def get_metric_tensor_and_sqrt_det_g_universal_autodecoder(autoencoder_cfg, cfg, charts, inverse=False):
    
    model = UniversalAutoencoder(cfg=autoencoder_cfg)
    decoder = ModulatedSIREN(cfg=autoencoder_cfg)
    model_apply_fn = model.apply


    init_points, supernode_idxs = jnp.zeros((16, autoencoder_cfg.dataset.num_points, 3)), jax.random.randint(jax.random.PRNGKey(0), (16, autoencoder_cfg.dataset.num_supernodes), 0, 128)
    params = model.init(jax.random.PRNGKey(0), init_points, supernode_idxs)["params"]
    optimizer = optax.adam(learning_rate=0.1)
    terget = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    state = checkpoints.restore_checkpoint(Path(cfg.autoencoder_checkpoint.checkpoint_path).absolute(), step=cfg.autoencoder_checkpoint.step, target=terget)
    params = state.params


    conditionings = []
    coords = {}
    key = jax.random.PRNGKey(0)
    for chart_key in charts.keys():
        key, subkey = jax.random.split(key)
        supernode_idxs = jax.random.permutation(subkey, jnp.arange(charts[chart_key].shape[0]))[:cfg.num_supernodes]
        out, coord, conditioning = model_apply_fn({"params": params}, charts[chart_key][None, :, :], supernode_idxs[None, :])
        conditionings.append(conditioning)
        coords[chart_key] = coord[0, :, :]
    
    with open(cfg.dataset.charts_path + "/charts2d.pkl", "wb") as f:
        pickle.dump(coords, f)

    conditionings = jnp.concatenate(conditionings, axis=0)
    d_params = params["siren"]


    def induced_riemannian_metric(conditioning, z):
        phi = lambda x, conditioning: decoder.apply({"params": d_params}, x, conditioning)
        J = jax.vmap(jax.jacfwd(phi, argnums=0), (0, None))(z, conditioning)[:, 0, :, :]
        return J.transpose(0, 2, 1) @ J

    def induced_inverse_riemannian_metric(conditioning, z):
        phi = lambda x, conditioning: decoder.apply({"params": d_params}, x, conditioning)
        J = jax.vmap(jax.jacfwd(phi, argnums=0), (0, None))(z, conditioning)[:, 0, :, :]
        return jnp.linalg.inv(J.transpose(0, 2, 1) @ J)

    def sqrt_det_g(conditioning, z):
        return jnp.sqrt(jnp.linalg.det(induced_riemannian_metric(conditioning, z)))

    if inverse:
        return (
            jax.jit(induced_inverse_riemannian_metric),
            jax.jit(sqrt_det_g),
            decoder,
        ), (conditionings, d_params)
    else:
        return (
            jax.jit(induced_riemannian_metric),
            jax.jit(sqrt_det_g),
            decoder,
        ), (conditionings, d_params)





def compute_norm_g_ginv_from_params(params, decoder_fn, noise_scale=0.1):

    rng_key = jax.random.PRNGKey(0)
    original_points = params["points"]
    num_points, dim = original_points.shape
    num_noisy = 10

    noise = jax.random.normal(rng_key, shape=(num_points, num_noisy, dim)) * noise_scale

    noisy_points = original_points[:, None, :] + noise
    noisy_points = noisy_points.reshape(-1, dim)
    all_points = jnp.concatenate([original_points, noisy_points], axis=0)

    recon_noisy_chart = decoder_fn({"params": params["D"]}, all_points)
    recon_chart = decoder_fn({"params": params["D"]}, original_points)
    latent_chart = params["points"]
    noisy_latent_chart = jnp.concatenate([params["points"], noisy_points], axis=0)

    d = lambda x: decoder_fn({"params": params["D"]}, x)
    J = vmap(jax.jacfwd(d))(all_points)
    J_T = jnp.transpose(J, (0, 2, 1))
    g = jnp.matmul(J_T, J)
    g_inv = jnp.linalg.inv(g)
    norm_g = jnp.linalg.norm(g, axis=(1, 2))
    norm_g_inv = jnp.linalg.norm(g_inv, axis=(1, 2))

    return (
        recon_noisy_chart,
        recon_chart,
        latent_chart,
        noisy_latent_chart,
        g,
        g_inv,
        norm_g,
        norm_g_inv,
    )


def compute_norm_g_ginv_from_params_autoencoder(
    model_fn, decoder_fn, chart, noise_scale=0.1
):

    rng_key = jax.random.PRNGKey(0)
    _, original_points = model_fn(chart)
    num_points, dim = original_points.shape
    num_noisy = 10

    noise = jax.random.normal(rng_key, shape=(num_points, num_noisy, dim)) * noise_scale

    noisy_points = original_points[:, None, :] + noise
    noisy_points = noisy_points.reshape(-1, dim)
    all_points = jnp.concatenate([original_points, noisy_points], axis=0)

    recon_noisy_chart = decoder_fn(all_points)
    recon_chart = decoder_fn(original_points)

    latent_chart = original_points
    noisy_latent_chart = jnp.concatenate([original_points, noisy_points], axis=0)

    d = lambda x: decoder_fn(x)
    J = vmap(jax.jacfwd(d))(all_points)
    J_T = jnp.transpose(J, (0, 2, 1))
    g = jnp.matmul(J_T, J)
    g_inv = jnp.linalg.inv(g)
    norm_g = jnp.linalg.norm(g, axis=(1, 2))
    norm_g_inv = jnp.linalg.norm(g_inv, axis=(1, 2))

    return (
        recon_noisy_chart,
        recon_chart,
        latent_chart,
        noisy_latent_chart,
        g,
        g_inv,
        norm_g,
        norm_g_inv,
    )
