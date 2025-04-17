import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm


def generate_sphere_mesh(radius=1.0, density=0, output_file=None):
    """
    Generate a sphere mesh with specified radius and density.
    
    Args:
        radius (float): The radius of the sphere
        density (int): The level of subdivision (0 = icosahedron, higher values give denser meshes)
        output_file (str, optional): Path to save the generated .obj file
                                    If None, the mesh vertices and connectivity are returned without saving
    
    Returns:
        tuple: (vertices, connectivity) of the generated sphere mesh
              or path to the saved .obj file if output_file is provided
    """
    # Start with an icosahedron
    verts, faces = create_icosahedron()
    
    # Subdivide the mesh to increase density
    for _ in range(density):
        verts, faces = subdivide(verts, faces)
    
    # Normalize vertices to lie on unit sphere
    lengths = np.sqrt(np.sum(verts**2, axis=1))
    verts = verts / lengths[:, np.newaxis]
    
    # Scale by radius
    verts = verts * radius
    
    # Save to file if output_file is specified
    if output_file is not None:
        save_obj(verts, faces, output_file)
    
    return verts, faces

def create_icosahedron():
    """
    Create an icosahedron (a polyhedron with 20 triangular faces).
    
    Returns:
        tuple: (vertices, faces) of the icosahedron
    """
    # Golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Vertices of the icosahedron
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=np.float32)
    
    # Normalize
    lengths = np.sqrt(np.sum(verts**2, axis=1))
    verts = verts / lengths[:, np.newaxis]
    
    # Faces (triangles)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)
    
    return verts, faces

def subdivide(verts, faces):
    """
    Subdivide each triangular face into 4 triangles.
    
    Args:
        verts (numpy.ndarray): Vertices array
        faces (numpy.ndarray): Faces array
    
    Returns:
        tuple: (new_vertices, new_faces)
    """
    # Create midpoints dictionary to store new vertices
    midpoints = {}
    new_faces = []
    new_midpoint_vertices = []
    
    # For each face, create three new vertices at the midpoints of each edge
    for face in faces:
        v1, v2, v3 = face
        
        # Get or create midpoints for each edge
        m1 = get_midpoint(midpoints, verts, v1, v2, new_midpoint_vertices)
        m2 = get_midpoint(midpoints, verts, v2, v3, new_midpoint_vertices)
        m3 = get_midpoint(midpoints, verts, v3, v1, new_midpoint_vertices)
        
        # Create four new triangular faces
        new_faces.append([v1, m1, m3])
        new_faces.append([v2, m2, m1])
        new_faces.append([v3, m3, m2])
        new_faces.append([m1, m2, m3])
    
    # Combine original vertices with new midpoint vertices
    new_verts = np.vstack([verts, np.array(new_midpoint_vertices)])
    
    return new_verts, np.array(new_faces)

def get_midpoint(midpoints, verts, v1, v2, new_midpoint_vertices):
    """
    Get the index of the midpoint between two vertices. Create if not exists.
    
    Args:
        midpoints (dict): Dictionary mapping edge key to midpoint index
        verts (numpy.ndarray): Vertices array
        v1, v2 (int): Indices of the vertices defining the edge
        new_midpoint_vertices (list): List to store the new midpoint vertices
    
    Returns:
        int: Index of the midpoint vertex
    """
    # Sort vertices to create a consistent key
    if v1 > v2:
        v1, v2 = v2, v1
    
    # Check if midpoint already exists
    key = (v1, v2)
    if key in midpoints:
        return midpoints[key]
    
    # Calculate midpoint position (simply average the positions)
    pos = (verts[v1] + verts[v2]) / 2.0
    
    # Normalize to lie on the unit sphere
    pos = pos / np.sqrt(np.sum(pos**2))
    
    # Add the new vertex to the list and store its index
    index = len(verts) + len(new_midpoint_vertices)
    midpoints[key] = index
    new_midpoint_vertices.append(pos)
    
    return index

def save_obj(verts, faces, filename):
    """
    Save vertices and faces to an OBJ file.
    
    Args:
        verts (numpy.ndarray): The vertices array
        faces (numpy.ndarray): The faces array
        filename (str): The path to save the file to
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory) and directory:
        os.makedirs(directory)
        
    with open(filename, 'w') as f:
        f.write("# Generated sphere mesh\n")
        
        # Write vertices
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces (add 1 because OBJ format uses 1-based indices)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def create_sphere_dataset(radius=1.0, density=2, points_per_unit_area=2, subset_cardinality=None, 
                          seed=42, center=True, output_dir="./datasets/sphere"):
    """
    Create a sphere dataset with a custom radius and density.
    
    Args:
        radius (float): The radius of the sphere
        density (int): The level of subdivision (0 = icosahedron, higher values give denser meshes)
        points_per_unit_area (float): Number of points to sample per unit area
        subset_cardinality (int, optional): Number of points to sample in total
        seed (int): Random seed for point sampling
        center (bool): Whether to center the dataset at the origin
        output_dir (str): Directory to save the generated .obj file
    
    Returns:
        datasets.sphere.Sphere: A Sphere dataset object
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the sphere mesh
    output_file = os.path.join(output_dir, f"sphere_r{radius}_d{density}.obj")
    generate_sphere_mesh(radius=radius, density=density, output_file=output_file)
    
    # Create and return the Sphere dataset
    from datasets.sphere import Sphere
    return Sphere(
        path=output_file,
        scale=1.0,  # No need to scale as we already set the radius
        points_per_unit_area=points_per_unit_area,
        subset_cardinality=subset_cardinality,
        seed=seed,
        center=center
    )

def visualize_geodesic_distances(vertices, faces, radius, density):
    """
    Generate a sphere mesh and visualize the geodesic distances from the north pole
    
    Args:
        radius (float): Radius of the sphere
        density (int): Density level of the sphere mesh
    """
    # Generate the sphere mesh
    
    # Calculate geodesic distances from the north pole (0, 0, radius)
    distances = geodesic_distances_from_pole_vectorized(vertices, radius)
    
    # Maximum distance is π*radius (from pole to pole)
    max_distance = np.pi * radius
    
    # Create a color map based on distances (normalized to [0, 1])
    normalized_distances = distances / max_distance
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the triangular mesh with colors based on distances
    cmap = cm.viridis
    colors = cmap(normalized_distances)
    
    # Create the triangular mesh with face colors
    triang = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                             triangles=faces, cmap=cmap)
    
    # Add a scatter plot of vertices with colors based on distances
    scatter = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                         c=distances, cmap=cmap, s=5, alpha=0.5)
    
    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Geodesic Distance from North Pole')
    
    # Highlight the north pole
    north_pole = np.array([[0, 0, radius]])
    ax.scatter(north_pole[:, 0], north_pole[:, 1], north_pole[:, 2], 
               color='red', s=100, marker='*', label='North Pole')
    
    # Add equator points for reference
    theta = np.linspace(0, 2*np.pi, 100)
    equator_x = radius * np.cos(theta)
    equator_y = radius * np.sin(theta)
    equator_z = np.zeros_like(theta)
    ax.plot(equator_x, equator_y, equator_z, color='black', linestyle='--', linewidth=1)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"Geodesic Distances on Sphere (r={radius}, density={density})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a legend
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"datasets/sphere/sphere_geodesic_distances_r{radius}_d{density}.png")
    plt.show()
    
    return fig, ax

def visualize_geodesic_contours(radius=1.0, density=3):
    """
    Generate a sphere mesh and visualize the geodesic distances as contour lines
    
    Args:
        radius (float): Radius of the sphere
        density (int): Density level of the sphere mesh
    """
    # Generate the sphere mesh
    vertices, faces = generate_sphere_mesh(radius=radius, density=density)
    
    # Calculate geodesic distances from the north pole (0, 0, radius)
    distances = geodesic_distances_from_pole_vectorized(vertices, radius)
    
    # Maximum distance is π*radius (from pole to pole)
    max_distance = np.pi * radius
    
    # Create contour levels (e.g., every 15 degrees)
    contour_levels = np.linspace(0, max_distance, 13)  # 0 to 180 degrees in 15-degree increments
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the base triangular mesh with a light color
    base_mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                                triangles=faces, color='lightgray', alpha=0.3)
    
    # Define custom contour colors
    contour_colors = plt.cm.plasma(np.linspace(0, 1, len(contour_levels)-1))
    
    # Plot contour lines for each level
    for i, level in enumerate(contour_levels[:-1]):
        # Find vertices close to this contour level
        mask = np.logical_and(distances >= level, distances < contour_levels[i+1])
        if np.any(mask):
            ax.scatter(vertices[mask, 0], vertices[mask, 1], vertices[mask, 2],
                      color=contour_colors[i], s=5, alpha=0.8, 
                      label=f"{level/np.pi*180:.0f}° - {contour_levels[i+1]/np.pi*180:.0f}°")
    
    # Highlight the north pole
    north_pole = np.array([[0, 0, radius]])
    ax.scatter(north_pole[:, 0], north_pole[:, 1], north_pole[:, 2], 
               color='red', s=100, marker='*', label='North Pole')
    
    # Add equator for reference
    theta = np.linspace(0, 2*np.pi, 100)
    equator_x = radius * np.cos(theta)
    equator_y = radius * np.sin(theta)
    equator_z = np.zeros_like(theta)
    ax.plot(equator_x, equator_y, equator_z, color='black', linestyle='--', linewidth=1, label='Equator')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"Geodesic Distance Contours on Sphere (r={radius})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a legend (with only key entries)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = ["North Pole", "Equator", "0° - 15°", "75° - 90°", "165° - 180°"]
    unique_handles = [handles[labels.index(label)] for label in unique_labels if label in labels]
    ax.legend(unique_handles, unique_labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"datasets/sphere/sphere_geodesic_contours_r{radius}_d{density}.png")
    plt.show()
    
    return fig, ax

def compare_analytical_vs_mesh_geodesic():
    """
    Compare analytical geodesic distances with those computed on meshes of different densities
    """
    # Parameters
    radius = 1.0
    densities = [0, 1, 2, 3, 4]  # Different mesh densities
    num_test_points = 100
    
    # Generate test points uniformly on the sphere
    np.random.seed(42)
    phi = np.random.uniform(0, 2*np.pi, num_test_points)
    cos_theta = np.random.uniform(-1, 1, num_test_points)
    theta = np.arccos(cos_theta)
    
    # Convert to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    test_points = np.column_stack((x, y, z))
    
    # Calculate true geodesic distances analytically
    true_distances = geodesic_distances_from_pole_vectorized(test_points, radius)
    
    # Calculate errors for different mesh densities
    results = []
    for density in densities:
        # Generate mesh
        vertices, faces = generate_sphere_mesh(radius=radius, density=density)
        
        # For each test point, find the closest mesh vertex
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices)
        _, closest_indices = tree.query(test_points)
        
        # Get the coordinates of the closest vertices
        closest_vertices = vertices[closest_indices]
        
        # Calculate geodesic distances for these closest vertices
        mesh_distances = geodesic_distances_from_pole_vectorized(closest_vertices, radius)
        
        # Calculate errors
        abs_errors = np.abs(mesh_distances - true_distances)
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        
        results.append({
            'density': density,
            'num_vertices': len(vertices),
            'max_error': max_error,
            'mean_error': mean_error
        })
        
    # Create plot of errors vs mesh density
    fig, ax = plt.subplots(figsize=(10, 6))
    densities = [r['density'] for r in results]
    max_errors = [r['max_error'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    
    ax.plot(densities, max_errors, 'o-', label='Maximum Error')
    ax.plot(densities, mean_errors, 's-', label='Mean Error')
    
    # Add vertex counts as text
    for i, r in enumerate(results):
        ax.annotate(f"{r['num_vertices']} vertices", 
                   (densities[i], max_errors[i]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
    
    ax.set_xlabel('Mesh Density')
    ax.set_ylabel('Geodesic Distance Error')
    ax.set_title('Errors in Geodesic Distance Computation vs Mesh Density')
    ax.set_xticks(densities)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("datasets/sphere/geodesic_errors_vs_density.png")
    plt.yscale('log')
    plt.show()
    
    # Print results
    print("Comparison of Analytical vs Mesh-based Geodesic Distances:")
    print("-" * 80)
    print(f"{'Density':<8} {'Vertices':<10} {'Max Error':<15} {'Mean Error':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['density']:<8} {r['num_vertices']:<10} {r['max_error']:<15.8f} {r['mean_error']:<15.8f}")
    
    return results

def geodesic_distance(p1, p2, radius=1.0):
    """
    Compute the exact geodesic distance between two points on a sphere.
    
    Args:
        p1 (numpy.ndarray): First point coordinates (x, y, z)
        p2 (numpy.ndarray): Second point coordinates (x, y, z)
        radius (float): Radius of the sphere
        
    Returns:
        float: Geodesic distance between the points along the sphere surface
    """
    # Normalize the points to ensure they are exactly on the sphere
    p1_norm = p1 / np.linalg.norm(p1) * radius
    p2_norm = p2 / np.linalg.norm(p2) * radius
    
    # Compute the central angle using the dot product
    # cos(theta) = (p1 · p2) / (|p1| * |p2|)
    cos_theta = np.clip(np.dot(p1_norm, p2_norm) / (radius * radius), -1.0, 1.0)
    
    # Calculate the central angle
    theta = np.arccos(cos_theta)
    
    # Geodesic distance = radius * central angle
    distance = radius * theta
    
    return distance

def geodesic_distances_from_pole(points, radius=1.0):
    """
    Compute the geodesic distances from the north pole (0, 0, r) to all given points.
    
    Args:
        points (numpy.ndarray): Array of point coordinates, shape (n, 3)
        radius (float): Radius of the sphere
        
    Returns:
        numpy.ndarray: Array of geodesic distances
    """
    # Define the north pole
    north_pole = np.array([0, 0, radius])
    
    # Initialize array for distances
    distances = np.zeros(len(points))
    
    # Calculate distance to each point
    for i, point in enumerate(points):
        distances[i] = geodesic_distance(north_pole, point, radius)
    
    return distances

def geodesic_distance_from_pole_analytic(point, radius=1.0):
    """
    Compute the geodesic distance from the north pole (0, 0, r) to a point analytically.
    This is a simpler and more efficient calculation specific to the north pole case.
    
    Args:
        point (numpy.ndarray): Point coordinates (x, y, z)
        radius (float): Radius of the sphere
        
    Returns:
        float: Geodesic distance from the north pole to the point
    """
    # Normalize the point to ensure it's exactly on the sphere
    point_norm = point / np.linalg.norm(point) * radius
    
    # For the north pole (0, 0, r), the central angle is directly related to the z-coordinate
    # cos(θ) = z / r
    cos_theta = np.clip(point_norm[2] / radius, -1.0, 1.0)
    
    # Calculate the central angle
    theta = np.arccos(cos_theta)
    
    # Geodesic distance = radius * central angle
    distance = radius * theta
    
    return distance

def geodesic_distances_from_pole_vectorized(points, radius=1.0):
    """
    Compute the geodesic distances from the north pole (0, 0, r) to all given points
    in a vectorized manner for efficiency.
    
    Args:
        points (numpy.ndarray): Array of point coordinates, shape (n, 3)
        radius (float): Radius of the sphere
        
    Returns:
        numpy.ndarray: Array of geodesic distances
    """
    # Normalize the points to ensure they are exactly on the sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points_norm = points / norms * radius
    
    # For the north pole (0, 0, r), we only need the z-coordinate
    cos_theta = np.clip(points_norm[:, 2] / radius, -1.0, 1.0)
    
    # Calculate the central angles
    theta = np.arccos(cos_theta)
    
    # Geodesic distances = radius * central angles
    distances = radius * theta
    
    return distances

def spherical_coordinates(points, radius=1.0):
    """
    Convert Cartesian coordinates to spherical coordinates (radius, theta, phi)
    where theta is the polar angle (from z-axis) and phi is the azimuthal angle.
    
    Args:
        points (numpy.ndarray): Array of point coordinates in Cartesian (x, y, z), shape (n, 3)
        radius (float): Radius of the sphere
        
    Returns:
        tuple: (r, theta, phi) arrays where:
            - r is the radial distance (should be constant = radius)
            - theta is the polar angle (from z-axis, 0 to π)
            - phi is the azimuthal angle (0 to 2π)
    """
    # Extract coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Calculate spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # Polar angle (from z-axis)
    phi = np.arctan2(y, x)  # Azimuthal angle
    
    return r, theta, phi


if __name__ == "__main__":
    
    radius = 4.0
    density = 4

    output_file = f"./datasets/sphere/sphere_r{int(radius)}_d{density}.obj"
    vertices, faces = generate_sphere_mesh(radius=radius, density=density, output_file=output_file)
    print(f"Generated sphere mesh saved to {output_file}")

    # Example 1: Visualize geodesic distances on a sphere
    print("Visualizing geodesic distances...")
    visualize_geodesic_distances(vertices, faces, radius, density)
    
    # Example 2: Visualize geodesic contours
    print("Visualizing geodesic contours...")
    visualize_geodesic_contours(radius, density)
    
    # Example 3: Compare analytical vs mesh-based geodesic distances
    print("Comparing analytical vs mesh-based geodesic distances...")
    compare_analytical_vs_mesh_geodesic()
    
    print("Examples complete. Check the generated images.") 