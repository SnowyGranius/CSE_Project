
import numpy as np
import matplotlib.pyplot as plt

# Data
shapes = ["circle", "rectangle", "ellipse", "triangle"]
M0_measured_values = [0.88739624, 0.92824779, 0.94359402, 0.95665847]
M0_theoretical_values = [0.8876751204372201, 0.92831114591458, 0.9436870855472399, 0.95670243022806]

M1_measured_values = [47.511876496565236, 41.88073935477212, 36.5726937971329, 37.79775785240023]
M1_theoretical_values = [44.773683110127, 40.362048941883, 34.654736584218995, 36.561243576213]

M2_measured_values = [1431, 1442, 1439, 1443]
M2_theoretical_values = [1442, 1442, 1442, 1442]

# X-axis positions
x = np.arange(len(shapes))

# Calculate relative errors
relative_errors = np.abs((np.array(M0_measured_values) - np.array(M0_theoretical_values)) / np.array(M0_theoretical_values))

# Plotting
fig, ax = plt.subplots()

# Plot points
ax.scatter(x, M0_measured_values, color='b', label='Measured')
ax.scatter(x, M0_theoretical_values, color='r', label='Theoretical')

# Draw lines connecting measured and theoretical points
for i in range(len(x)):
    ax.plot([x[i], x[i]], [M0_measured_values[i], M0_theoretical_values[i]], color='gray', linestyle='--')

# Annotate relative errors in between the points
for i, (measured, theoretical, error) in enumerate(zip(M0_measured_values, M0_theoretical_values, relative_errors)):
    mid_point = (measured + theoretical) / 2
    ax.annotate(f'{error:.2%}', (x[i], mid_point), textcoords="offset points", xytext=(10,0), ha='left', color='black')

# Add labels, title, and legend
ax.set_xlabel('Shapes')
ax.set_ylabel('M0')
ax.set_title('M0 comparison measured vs theoretical')
ax.set_xticks(x)
ax.set_xticklabels(shapes)
ax.legend()

# Show plot
plt.show()