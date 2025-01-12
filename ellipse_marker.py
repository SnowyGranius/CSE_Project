import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle

# Define an ellipse as a custom marker
def create_ellipse_marker(a=1.0, b=0.5, num_points=100):
    """
    Create an ellipse path with aspect ratio a/b.

    Parameters:
        a (float): Semi-major axis length (horizontal).
        b (float): Semi-minor axis length (vertical).
        num_points (int): Number of points to approximate the ellipse.
    
    Returns:
        Path: A matplotlib Path object for the ellipse.
    """
    # Angles for the parametric equation of an ellipse
    theta = np.linspace(0, 2 * np.pi, num_points)
    # Parametric equation for ellipse
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    # Combine x and y into vertices
    vertices = np.column_stack([x, y])

    # Close the path
    vertices = np.append(vertices, [vertices[0]], axis=0)

    # Path codes
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]

    return Path(vertices, codes)

# Create a custom ellipse marker with a/b = 0.5
ellipse_path = create_ellipse_marker(a=1.0, b=0.5)
ellipse = MarkerStyle(ellipse_path)