import os
import tarfile

import numpy as np
import open3d as o3d

wd = os.path.dirname(os.path.abspath(__file__))

tar_file_path = wd + "/bunny.tar.gz"
extract_folder = wd + "/bunny/reconstruction"

with tarfile.open(tar_file_path, "r:gz") as tar:
    tar.extractall(path=extract_folder)

ply_file_path = f"{extract_folder}/bun_zipper.ply"

pcd = o3d.io.read_point_cloud(ply_file_path)

# Convert point cloud to mesh using Poisson reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)


# Sample points from the mesh
def sample_points_from_mesh(mesh, points_per_unit_area=40):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    all_points = [vertices]

    for triangle in triangles:
        v1, v2, v3 = vertices[triangle]

        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        area = np.linalg.norm(normal) / 2

        num_samples = max(1, int(area * points_per_unit_area))

        r1 = np.random.random((num_samples, 1))
        r2 = np.random.random((num_samples, 1))

        mask = (r1 + r2) > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]

        a = 1 - r1 - r2
        b = r1
        c = r2

        points = a * v1 + b * v2 + c * v3
        all_points.append(points)

    return np.vstack(all_points)


# Sample additional points
points = sample_points_from_mesh(mesh, points_per_unit_area=40)
np.save(wd + "/stanford_bunny.npy", points)

# Visualize the sampled points
pcd_sampled = o3d.geometry.PointCloud()
pcd_sampled.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd_sampled])
