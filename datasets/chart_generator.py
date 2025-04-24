import numpy as np
from scipy.interpolate import RBFInterpolator

def generate_disk_pointcloud(n_points=1000, radius=1.0):
    # Generate random points in polar coordinates
    r = np.sqrt(np.random.uniform(0, radius**2, n_points))
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)  # Flat disk in z=0 plane
    
    return np.column_stack((x, y, z))

def random_smooth_deformation(points, n_control_points=20, deformation_scale=0.3):
    """Apply a smooth random deformation using Radial Basis Functions"""
    # Generate random control points around the disk
    control_points = np.random.uniform(-1.5, 1.5, (n_control_points, 3))
    
    # Generate random displacement vectors for control points
    displacements = np.random.normal(0, deformation_scale, (n_control_points, 3))
    
    # Create RBF interpolator for smooth deformation
    rbf = RBFInterpolator(control_points, displacements, kernel='thin_plate_spline')
    
    # Calculate displacement field for all points
    deformation = rbf(points)
    
    # Apply deformation
    deformed_points = points + deformation
    
    return deformed_points

def generate_deformed_disk(n_points=1000, deformation_scale=0.3):
    """Generate a disk point cloud and apply random smooth deformation"""
    # Generate initial disk
    points = generate_disk_pointcloud(n_points)
    
    # Apply deformation
    deformed_points = random_smooth_deformation(points, deformation_scale=deformation_scale)
    
    return deformed_points

# Example usage:
if __name__ == "__main__":
    # Generate deformed disk
    points = generate_deformed_disk(n_points=1000, deformation_scale=0.3)
    
    # Optionally visualize using matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('deformed_disk.png')
    plt.show()
