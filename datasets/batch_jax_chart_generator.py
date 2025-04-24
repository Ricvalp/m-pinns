import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.offline import plot
from functools import partial
from jax import random
from jax import jit
import time # For timing

# Configure JAX
jax.config.update('jax_enable_x64', True)

# --- RBF Kernel Functions (Unchanged) ---
# @jit
def multiquadric_kernel(r, epsilon):
    return jnp.sqrt(1.0 + (epsilon * r)**2)
# @jit
def inverse_multiquadric_kernel(r, epsilon):
    return 1.0 / jnp.sqrt(1.0 + (epsilon * r)**2)
# @jit
def gaussian_kernel(r, epsilon):
    return jnp.exp(-(epsilon * r)**2)
# @jit
def linear_kernel(r, epsilon):
    return r
# @jit
def thin_plate_spline_kernel(r, epsilon):
    return jnp.where(r == 0, 0.0, r**2 * jnp.log(r))


# --- RBF Core Implementation (Unchanged) ---
@partial(jit, static_argnames=['kernel_func'])
def fit_rbf(control_points, displacements, kernel_func, epsilon, reg=1e-8):
    n_control, n_dims = control_points.shape
    diff = control_points[:, None, :] - control_points[None, :, :]
    distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    A = kernel_func(distances, epsilon)
    A += jnp.eye(n_control) * reg
    weights = jnp.linalg.solve(A, displacements)
    return weights

@partial(jit, static_argnames=['kernel_func'])
def evaluate_rbf(points_to_eval, control_points, weights, kernel_func, epsilon):
    diff = points_to_eval[:, None, :] - control_points[None, :, :]
    distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    Phi_eval = kernel_func(distances, epsilon)
    delta_points = Phi_eval @ weights
    return delta_points

# --- Main Deformation Function (JIT applied, Unchanged Logic) ---
@partial(jit, static_argnums=(2, 4)) # n_control_points, kernel_func are static
def rbf_deformation_jax(key, points, n_control_points, deformation_scale, kernel_func, epsilon, control_point_range=1.5, reg=1e-8):
    key, subkey1, subkey2 = random.split(key, 3)
    n_dims = points.shape[1]
    control_points = random.uniform(subkey1,
                                    (n_control_points, n_dims),
                                    minval=-control_point_range,
                                    maxval=control_point_range)
    displacements = random.normal(subkey2, (n_control_points, n_dims)) * deformation_scale
    weights = fit_rbf(control_points, displacements, kernel_func, epsilon, reg)
    delta_points = evaluate_rbf(points, control_points, weights, kernel_func, epsilon)
    deformed_points = points + delta_points
    return deformed_points


# --- Utility Functions (Disk Generation, Rotation - Unchanged Logic) ---
# No JIT needed here as it's simple setup, JIT happens in deformation/rotation
def generate_disk_pointcloud(key, n_points, radius=1.0):
    key, subkey1, subkey2 = random.split(key, 3)
    r = jnp.sqrt(random.uniform(subkey1, (n_points,), minval=0, maxval=radius**2))
    theta = random.uniform(subkey2, (n_points,), minval=0, maxval=2*jnp.pi)
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    z = jnp.zeros_like(x)
    return jnp.column_stack((x, y, z))

@jit
def random_rotation_matrix(key):
    key, subkey1, subkey2 = random.split(key, 3)
    axis = random.normal(subkey1, (3,))
    axis_norm = jnp.linalg.norm(axis)
    axis = jnp.where(axis_norm > 1e-6, axis / axis_norm, jnp.array([1.0, 0.0, 0.0]))
    theta = random.uniform(subkey2, (1,)) * 2.0 * jnp.pi
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    K = jnp.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    R = jnp.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)
    return R.squeeze()

# --- Single Disk Generation Function ---
# This function contains the logic for generating ONE deformed disk.
# We keep it separate so we can easily vmap it.
def generate_deformed_disk_rbf_single(key, n_points, radius, n_control_points, deformation_scale, kernel_func, epsilon, control_point_range, reg):
    """Generates ONE disk, deforms it using JAX RBF, and rotates it."""
    key, subkey1, subkey2, subkey3 = random.split(key, 4)

    # 1. Generate initial flat disk
    points = generate_disk_pointcloud(subkey1, n_points, radius=radius)

    # 2. Apply JAX RBF deformation
    # Pass static args correctly
    deformed_points = rbf_deformation_jax(subkey2, points, n_control_points,
                                          deformation_scale, kernel_func, epsilon,
                                          control_point_range, reg)

    # 3. Generate random rotation
    rotation = random_rotation_matrix(subkey3)

    # 4. Apply rotation
    rotated_deformed_points = deformed_points @ rotation.T

    return rotated_deformed_points

# --- Batched Disk Generation using vmap ---
# Create the batched version by mapping over the 'key' argument (axis 0).
# All other arguments are treated as static (in_axes=None).
# Note: kernel_func is static *within* rbf_deformation_jax, but here it's treated
# like any other constant argument passed to each vmapped call.
batched_generate_disks = jax.vmap(
     ,
    in_axes=(0, None, None, None, None, None, None, None, None)
    # Axis specifications for arguments:
    # key: 0 (map over keys)
    # n_points: None (broadcast) 
    # n_control_points: None (broadcast)
    # deformation_scale: None (broadcast)
    # kernel_func: None (broadcast - the function itself is passed)
    # epsilon: None (broadcast)
    # control_point_range: None (broadcast)
    # reg: None (broadcast)
)


# --- Main Execution ---
if __name__ == "__main__":
    # --- Parameters ---
    seed = 2024
    batch_size = 128          # <<< Number of disks to generate in the batch
    num_points = 5000       # Points per disk
    disk_radius = 1.0
    num_control = 10        # Control points per disk
    deform_scale = 0.5     # Max displacement scale
    rbf_kernel = gaussian_kernel # Choose kernel
    kernel_epsilon = 2.5    # Kernel shape parameter
    rbf_regularization = 1e-7 # Regularization

    # --- Generation ---
    main_key = random.PRNGKey(seed)

    # Generate a batch of keys, one for each disk
    keys = random.split(main_key, batch_size)

    print(f"Generating batch of {batch_size} deformed disks ({num_points} points each)...")
    print(f"Kernel: {rbf_kernel.__name__}, Epsilon: {kernel_epsilon}, Controls: {num_control}")

    batched_generate_disks(
        keys,                 # The batch of keys
        num_points,           # Constant for all disks
        disk_radius,          # Constant
        num_control,          # Constant
        deform_scale,         # Constant
        rbf_kernel,           # Constant (the function object)
        kernel_epsilon,       # Constant
        1.5, # control_point_range (Constant)
        rbf_regularization    # Constant
    )

    # Time the JIT compilation + execution
    start_time = time.time()
    # Call the vmap'ped function with the batch of keys
    deformed_disks_batch = batched_generate_disks(
        keys,                 # The batch of keys
        num_points,           # Constant for all disks
        disk_radius,          # Constant
        num_control,          # Constant
        deform_scale,         # Constant
        rbf_kernel,           # Constant (the function object)
        kernel_epsilon,       # Constant
        1.5, # control_point_range (Constant)
        rbf_regularization    # Constant
    ).block_until_ready()     # Ensure computation finishes for timing
    end_time = time.time()

    print(f"Batch generation took {end_time - start_time:.4f} seconds")
    print(f"Output shape: {deformed_disks_batch.shape}") # Should be (batch_size, num_points, 3)

    # --- Visualization ---
    print("Preparing visualization (plotting first 16 disks)...")
    fig = go.Figure()

    # Plot the first 16 disks from the batch
    num_disks_to_plot = min(16, batch_size)
    
    for i in range(num_disks_to_plot):
        disk_to_plot = deformed_disks_batch[i]
        
        x_coords = jnp.asarray(disk_to_plot[:, 0])
        y_coords = jnp.asarray(disk_to_plot[:, 1])
        z_coords = jnp.asarray(disk_to_plot[:, 2])
        
        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(
                size=2.0,
                opacity=0.7,
                color=z_coords,
                colorscale='Viridis',
            ),
            name=f'Disk {i}'
        ))

    # Update layout
    max_coord = disk_radius + deform_scale * 1.5
    fig.update_layout(
        title=f'Batched JAX RBF Deformation (First {num_disks_to_plot} Disks)',
        scene=dict(
            xaxis=dict(range=[-max_coord, max_coord], title='X'),
            yaxis=dict(range=[-max_coord, max_coord], title='Y'),
            zaxis=dict(range=[-max_coord, max_coord], title='Z'),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube'
        ),
        width=900, height=900,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    output_filename = 'jax_rbf_deformed_disk_batch.html'
    plot(fig, filename=output_filename, auto_open=True)
    print(f"Visualization saved as '{output_filename}'")