from absl import app, logging
import ml_collections
import wandb
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from torch.utils.data import Dataset
from pathlib import Path
from functools import partial
import jax
import optax
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import networkx as nx

from universal_autoencoder.upt_autoencoder import UniversalAutoencoder
from universal_autoencoder.siren import ModulatedSIREN

def load_cfgs():
    """Load configuration from file."""
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0
    cfg.figure_path = "figures/test_fit_universal_autoencoder"

    cfg.dataset = ml_collections.ConfigDict()
    cfg.dataset.num_points = 1000
    cfg.dataset.disk_radius = 1.0
    cfg.dataset.num_supernodes = 32
    cfg.dataset.nearest_neighbors_distance_matrix = 10

    cfg.train = ml_collections.ConfigDict()
    cfg.train.batch_size = 64
    cfg.train.lr = 1e-5
    cfg.train.num_steps = 50000
    cfg.train.reg = "geo+riemannian"
    cfg.train.noise_scale_riemannian = 0.01
    cfg.wandb = ml_collections.ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.wandb_log_every = 10

    cfg.profiler = ml_collections.ConfigDict()
    cfg.profiler.use = True
    cfg.profiler.log_dir = "universal_autoencoder/profiler/"
    cfg.profiler.start_step = 20
    cfg.profiler.end_step = 30

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # #   EncoderSupernodes # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.encoder_supernodes_cfg = ml_collections.ConfigDict()
    cfg.encoder_supernodes_cfg.max_degree = 5
    cfg.encoder_supernodes_cfg.input_dim = 3
    cfg.encoder_supernodes_cfg.gnn_dim = 64
    cfg.encoder_supernodes_cfg.enc_dim = 64
    cfg.encoder_supernodes_cfg.enc_depth = 4
    cfg.encoder_supernodes_cfg.enc_num_heads = 4
    cfg.encoder_supernodes_cfg.perc_dim = 64
    cfg.encoder_supernodes_cfg.perc_num_heads = 4
    cfg.encoder_supernodes_cfg.num_latent_tokens = None
    cfg.encoder_supernodes_cfg.init_weights = "truncnormal"
    cfg.encoder_supernodes_cfg.output_coord_dim = 2
    cfg.encoder_supernodes_cfg.coord_enc_dim = 64
    cfg.encoder_supernodes_cfg.coord_enc_depth = 2
    cfg.encoder_supernodes_cfg.coord_enc_num_heads = 4
    cfg.encoder_supernodes_cfg.latent_encoder_depth = 2
    cfg.encoder_supernodes_cfg.ndim = 3
    cfg.encoder_supernodes_cfg.perc_depth = 4

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  SIREN  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.modulated_siren_cfg = ml_collections.ConfigDict()
    cfg.modulated_siren_cfg.output_dim = 3
    cfg.modulated_siren_cfg.num_layers = 4
    cfg.modulated_siren_cfg.hidden_dim = 256
    cfg.modulated_siren_cfg.omega_0 = 30.0
    cfg.modulated_siren_cfg.modulation_hidden_dim = 256
    cfg.modulated_siren_cfg.modulation_num_layers = 4
    cfg.modulated_siren_cfg.shift_modulate = True
    cfg.modulated_siren_cfg.scale_modulate = False

    return cfg


def main(_):
    cfg = load_cfgs()

    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)
    
    if cfg.profiler.use:
        Path(cfg.profiler.log_dir).mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(cfg.seed)
    key, params_subkey, supernode_subkey = jax.random.split(key, 3)
    dataset=HalfSphereDataset(
            num_points=cfg.dataset.num_points, 
            num_supernodes=cfg.dataset.num_supernodes,
            nearest_neighbors_distance_matrix=cfg.dataset.nearest_neighbors_distance_matrix
        )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=numpy_collate,
    )

    run = None
    if cfg.wandb.use:
        run = wandb.init(
            project="test_universal_autoencoder",
            entity="ricvalp",
            name=f"test_universal_autoencoder",
            config=cfg.to_dict(),
        )
        wandb_id = run.id
    else:
        wandb_id = "no_wandb_" + str(cfg.seed)

    model = UniversalAutoencoder(cfg=cfg)
    decoder = ModulatedSIREN(cfg=cfg)
    decoder_apply_fn = decoder.apply

    decoder_apply_fn = model.apply

    @partial(jax.vmap, in_axes=(None, 0))
    def geodesic_preservation_loss(distances_matrix, z):
        z_diff = z[:, None, :] - z[None, :, :]
        z_dist = jnp.sqrt(jnp.sum(z_diff**2, axis=-1) + 1e-8)
        z_dist = z_dist / jnp.mean(z_dist)
        geodesic_dist = distances_matrix / jnp.mean(distances_matrix)
        return jnp.mean((z_dist - geodesic_dist) ** 2)

    def riemannian_metric_loss(params, latent, coords, key):
        d = lambda x: decoder_apply_fn({"params": params["siren"]}, x, latent)
        noise = (
            jax.random.normal(key, shape=coords.shape)
            * cfg.train.noise_scale_riemannian
        )
        coords = coords + noise
        J = jax.vmap(jax.jacfwd(d))(coords)
        J_T = jnp.transpose(J, (0, 2, 1))
        g = jnp.matmul(J_T, J)
        g_inv = jnp.linalg.inv(g)
        return jnp.mean(jnp.absolute(g)) + 0.1 * jnp.mean(jnp.absolute(g_inv))

    distance_matrix = dataset.distances_matrix

    if cfg.train.reg == "geo+riemannian":
        def loss_fn(params, batch, key):
            points, supernode_idxs = batch
            pred, coords, conditioning = state.apply_fn({"params": params}, points, supernode_idxs)
            recon_loss = jnp.sum((pred - points) ** 2, axis=-1).mean()
            return (recon_loss +
                    geodesic_preservation_loss(distance_matrix, coords).mean() + 
                    riemannian_metric_loss(params, conditioning, coords, key)
                    )

    if cfg.train.reg == "geodesic_preservation":
        def loss_fn(params, batch, key):
            points, supernode_idxs = batch
            pred, coords, conditioning = state.apply_fn({"params": params}, points, supernode_idxs)
            recon_loss = jnp.sum((pred - points) ** 2, axis=-1).mean()
            return recon_loss + geodesic_preservation_loss(distance_matrix, coords).mean()
        
    elif cfg.train.reg == "none":
        def loss_fn(params, batch, key):
            points, supernode_idxs = batch
            pred, coords, conditioning = state.apply_fn({"params": params}, points, supernode_idxs)
            recon_loss = jnp.sum((pred - points) ** 2, axis=-1).mean()
            return recon_loss
    
    else:
        raise ValueError(f"Invalid regularization method: {cfg.train.reg}")

    @jax.jit
    def train_step(state, batch, key):
        my_loss = lambda params: loss_fn(params, batch, key)
        loss, grads = jax.value_and_grad(my_loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, grads

    batch_size = cfg.train.batch_size
    num_points = cfg.dataset.num_points
    num_supernodes = cfg.encoder_supernodes_cfg.max_degree

    # Optional condition
    supernode_idxs = jax.random.randint(
        supernode_subkey, (batch_size, num_supernodes), 0, num_points
    )

    # Initialize parameters
    init_points, supernode_idxs = next(iter(data_loader))
    params = model.init(params_subkey, init_points, supernode_idxs)["params"]
    print(f"Number of parameters: {count_parameters(params)}")
    optimizer = optax.adam(learning_rate=cfg.train.lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    progress_bar = tqdm(range(cfg.train.num_steps))
    step = 0
    data_loader_iter = iter(data_loader)

    for _ in progress_bar:

        if cfg.profiler.use:
            set_profiler(cfg.profiler, step, cfg.profiler.log_dir)

        try:
            batch = next(data_loader_iter)
            key, subkey = jax.random.split(key)
            state, loss, grads = train_step(state, batch, subkey)

            if step % cfg.wandb.wandb_log_every == 0:

                log_dict = {
                    "loss": loss,
                    "step": step,
                }
                if cfg.wandb.use:
                    wandb.log(log_dict, step=step)

            progress_bar.set_postfix(loss=float(loss))

            step += 1

        except StopIteration:
            
            data_loader_iter = iter(data_loader)
    
    # Add test reconstruction after training
    print("Testing reconstruction...")
    test_mse = test_reconstruction(state, data_loader)
    
    if cfg.wandb.use:
        wandb.log({"final_reconstruction_mse": test_mse})
        # Upload the reconstruction image to wandb
        wandb.log({"reconstruction_samples": wandb.Image(f"figures/test_fit_universal_autoencoder/reconstruction_samples.png")})



def create_graph(
    pts,
    nearest_neighbors,
):
    """

    Create a graph from the points.

    Args:
        pts (np.ndarray): The points
        n (int): The number of nearest neighbors
        connectivity (np.ndarray): The connectivity of the mesh

    Returns:
        G (nx.Graph): The graph

    """

    # Create a n-NN graph
    tree = KDTree(pts)
    G = nx.Graph()

    # Add nodes to the graph
    for i, point in enumerate(pts):
        G.add_node(i, pos=point)

    # Add edges to the graph
    logging.info("Building the graph...")
    distances, indices = tree.query(
        pts, nearest_neighbors + 1
    )  # n+1 because the point itself is included

    for i in range(len(pts)):
        for j in range(
            1, nearest_neighbors + 1
        ):  # start from 1 to exclude the point itself
            neighbor_index = indices[i, j]
            distance = distances[i, j]
            G.add_edge(i, neighbor_index, weight=distance)

    logging.info(f"Graph created with {len(G.nodes)} nodes and {len(G.edges)} edges")

    return G


def calculate_distance_matrix_single_process(pts, nearest_neighbors):
    logging.info(f"Calculating distances for {len(pts)} points")
    G = create_graph(pts=pts, nearest_neighbors=nearest_neighbors)
    # Check that graph is a single connected component
    if not nx.is_connected(G):
        raise ValueError(
            f"The graph is not a single connected component"
        )
    # distances = dict(nx.all_pairs_shortest_path_length(G, cutoff=None))
    distances = dict(nx.all_pairs_dijkstra_path_length(G, cutoff=None))
    distances_matrix = np.zeros((len(pts), len(pts)))
    for j in range(len(pts)):
        for k in range(len(pts)):
            distances_matrix[j, k] = distances[j][k]
    logging.info(f"Finished calculating distances")
    return distances_matrix


class HalfSphereDataset(Dataset):
    def __init__(self, num_points, num_supernodes, nearest_neighbors_distance_matrix):
        self.num_points = num_points
        self.num_supernodes = num_supernodes
        theta = np.random.uniform(0, jnp.pi/2, (num_points, 1))
        phi = np.random.uniform(0, 2 * jnp.pi, (num_points, 1))

        # Generate points on a sphere
        self.points = np.concatenate(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        )

        self.distances_matrix = calculate_distance_matrix_single_process(self.points, nearest_neighbors_distance_matrix)

        # Generate random supernode indices
        self.random_supernode_idxs = []
        for i in range(10000):
            self.random_supernode_idxs.append(np.random.permutation(self.num_points)[: self.num_supernodes])

    def get_rotated_points(self):

        # Generate random rotation matrix
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, 2*np.pi) 
        psi = np.random.uniform(0, 2*np.pi)

        # Rotation matrix around x axis
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])

        # Rotation matrix around y axis  
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                      [0, 1, 0],
                      [-np.sin(phi), 0, np.cos(phi)]])

        # Rotation matrix around z axis
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Apply rotation
        return (R @ self.points.T).T

    def get_rotated_scaled_points(self):

        # Generate random rotation matrix
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, 2*np.pi) 
        psi = np.random.uniform(0, 2*np.pi)
        scale = np.random.uniform(0.5, 1.5)

        # Rotation matrix around x axis
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])

        # Rotation matrix around y axis  
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                      [0, 1, 0],
                      [-np.sin(phi), 0, np.cos(phi)]])

        # Rotation matrix around z axis
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Apply rotation
        return (scale * R @ self.points.T).T


    def __len__(self):
        return 100000000 # self.num_points

    def __getitem__(self, idx):
        supernode_idxs = self.random_supernode_idxs[np.random.randint(0, len(self.random_supernode_idxs))]
        # points = self.get_rotated_points()
        points = self.get_rotated_scaled_points()
        return points, supernode_idxs


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def set_profiler(profiler_config, step, log_dir):
    if profiler_config is not None:
        if step == profiler_config.start_step:
            jax.profiler.start_trace(log_dir=log_dir)
        if step == profiler_config.end_step:
            jax.profiler.stop_trace()


def count_parameters(params):
    """Count the number of parameters in a parameter tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def test_reconstruction(state, data_loader, num_samples=5):
    """Test the reconstruction ability of the model and visualize results in 3D.
    
    Args:
        state: The trained model state
        data_loader: DataLoader to get test samples
        num_samples: Number of samples to visualize
    """
    # Get a batch of data
    data_iter = iter(data_loader)
    points, supernode_idxs = next(data_iter)
    
    # Generate reconstructions
    reconstructions, coords, conditioning = state.apply_fn({"params": state.params}, points, supernode_idxs)
    
    # Convert to numpy for plotting
    points_np = np.array(points)
    reconstructions_np = np.array(reconstructions)
    coords_np = np.array(coords)
    
    # Visualize the first num_samples examples
    n = min(num_samples, len(points))
    fig = plt.figure(figsize=(15, 5*n))
    
    for i in range(n):
        # Original points
        ax1 = fig.add_subplot(n, 3, 3*i+1, projection='3d')
        ax1.scatter(
            points_np[i, :, 0], 
            points_np[i, :, 1], 
            points_np[i, :, 2], 
            c='blue', alpha=0.6, s=10
        )
        ax1.set_title(f'Original Points (Sample {i+1})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim([-1.1, 1.1])
        ax1.set_ylim([-1.1, 1.1])
        ax1.set_zlim([-1.1, 1.1])
        
        # Reconstructed points
        ax2 = fig.add_subplot(n, 3, 3*i+2, projection='3d')
        ax2.scatter(
            reconstructions_np[i, :, 0], 
            reconstructions_np[i, :, 1], 
            reconstructions_np[i, :, 2], 
            c='red', alpha=0.6, s=10
        )
        ax2.set_title(f'Reconstructed Points (Sample {i+1})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_xlim([-1.1, 1.1])
        ax2.set_ylim([-1.1, 1.1])
        ax2.set_zlim([-1.1, 1.1])
        
        # Learned coordinates
        ax3 = fig.add_subplot(n, 3, 3*i+3)
        scatter = ax3.scatter(
            coords_np[i, :, 0], 
            coords_np[i, :, 1],
            alpha=0.8,
            s=10
        )
        ax3.set_title(f'Learned 2D Coordinates (Sample {i+1})')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f"figures/test_fit_universal_autoencoder/reconstruction_samples.png", dpi=300)
    plt.close()
    
    # Calculate reconstruction error
    mse = np.mean(np.sum((points_np - reconstructions_np) ** 2, axis=-1))
    print(f"Reconstruction MSE: {mse:.6f}")
    
    return mse


if __name__ == "__main__":
    app.run(main)