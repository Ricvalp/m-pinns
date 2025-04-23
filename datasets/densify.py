import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datasets.utils import densify_mesh, Mesh


def plot_mesh(verts, connectivity, ax, color="lightblue", alpha=0.5, wireframe=True):
    """Plot a triangular mesh in 3D"""
    triangles = verts[connectivity]

    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=connectivity,
        color=color,
        alpha=alpha,
        shade=True,
    )

    # if wireframe:
    #     for triangle in connectivity:
    #         v0, v1, v2 = triangle
    #         x = verts[[v0, v1, v2, v0], 0]
    #         y = verts[[v0, v1, v2, v0], 1]
    #         z = verts[[v0, v1, v2, v0], 2]
    #         ax.plot(x, y, z, 'k-', linewidth=0.5)

    return ax


if __name__ == "__main__":

    obj_path = "./datasets/coil/"
    obj_name = "coil_1.2.obj"

    input_obj_file = obj_path + obj_name  # Replace with the path to your input OBJ file
    output_obj_file = (
        obj_path + "dense_" + obj_name
    )  # Name for the output denser mesh file
    subdivision_iterations = 1  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Increase this number for a denser mesh.
    # Be careful: faces/vertices grow exponentially!
    # 1 iteration: 4x faces
    # 2 iterations: 16x faces
    # 3 iterations: 64x faces
    # 4 iterations: 256x faces ...

    m = Mesh(input_obj_file)
    verts, connectivity = m.verts, m.connectivity

    # Densify the mesh
    densify_mesh(
        input_obj_file, output_obj_file, subdivision_iterations=subdivision_iterations
    )
    dense_mesh = Mesh(output_obj_file)
    dense_verts, dense_connectivity = dense_mesh.verts, dense_mesh.connectivity

    fig = plt.figure(figsize=(16, 10))

    ax = plt.figure(figsize=(10, 10)).add_subplot(projection="3d")
    surf = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], verts[:, 2], triangles=connectivity
    )
    plt.savefig(obj_path + "original_mesh.png")
    plt.close()

    ax = plt.figure(figsize=(10, 10)).add_subplot(projection="3d")
    surf = ax.plot_trisurf(
        dense_verts[:, 0],
        dense_verts[:, 1],
        dense_verts[:, 2],
        triangles=dense_connectivity,
    )
    plt.savefig(obj_path + "dense_mesh.png")
    plt.close()

    # Display the statistics
    print("Mesh Densification Statistics:")
    print(f"Original mesh: {len(verts)} vertices, {len(connectivity)} faces")
    print(f"Level 1: {len(dense_verts)} vertices, {len(dense_connectivity)} faces")
