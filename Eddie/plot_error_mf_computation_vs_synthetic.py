import numpy as np
import matplotlib.pyplot as plt
from synthetic_microstructure_test import array_creation

# Data
shapes = ["circle", "rectangle", "ellipse", "triangle"]
#M0_measured_values = [0.88739624, 0.92824779, 0.94359402, 0.95665847]
M0_theoretical_values = [0.8876751204372201, 0.92831114591458, 0.9436870855472399, 0.95670243022806]

#M2_measured_values = [47.511876496565236, 41.88073935477212, 36.5726937971329, 37.79775785240023]
M1_theoretical_values = [44.773683110127, 40.362048941883, 34.654736584218995, 36.561243576213]

#M2_measured_values = [1431, 1442, 1439, 1443]
M2_theoretical_values = [1442, 1442, 1442, 1442]

M0_measured_values, M1_measured_values, M2_measured_values = array_creation()

# X-axis positions
x = np.arange(1000, 11000, 1000)

absolute_errors_circle_M0 = np.abs(M0_measured_values[0] - np.array(M0_theoretical_values)[0])
absolute_errors_rectangle_M0 = np.abs(M0_measured_values[1] - np.array(M0_theoretical_values)[1])
absolute_errors_ellipse_M0 = np.abs(M0_measured_values[2] - np.array(M0_theoretical_values)[2])
absolute_errors_triangle_M0 = np.abs(M0_measured_values[3] - np.array(M0_theoretical_values)[3])

# Calculate relative errors
relative_errors_circle_M1 = np.abs((M1_measured_values[0] - np.array(M1_theoretical_values)[0]) / np.array(M1_theoretical_values)[0])*100
relative_errors_rectangle_M1 = np.abs((M1_measured_values[1] - np.array(M1_theoretical_values)[1]) / np.array(M1_theoretical_values)[1])*100
relative_errors_ellipse_M1 = np.abs((M1_measured_values[2] - np.array(M1_theoretical_values)[2]) / np.array(M1_theoretical_values)[2])*100
relative_errors_triangle_M1 = np.abs((M1_measured_values[3] - np.array(M1_theoretical_values)[3]) / np.array(M1_theoretical_values)[3])*100


absolute_errors_circle_M1 = np.abs(M1_measured_values[0] - np.array(M1_theoretical_values)[0])
absolute_errors_rectangle_M1 = np.abs(M1_measured_values[1] - np.array(M1_theoretical_values)[1])
absolute_errors_ellipse_M1 = np.abs(M1_measured_values[2] - np.array(M1_theoretical_values)[2])
absolute_errors_triangle_M1 = np.abs(M1_measured_values[3] - np.array(M1_theoretical_values)[3])

absolute_errors_circle_M2 = np.abs(M2_measured_values[0] - np.array(M2_theoretical_values)[0])
absolute_errors_rectangle_M2 = np.abs(M2_measured_values[1] - np.array(M2_theoretical_values)[1])
absolute_errors_ellipse_M2 = np.abs(M2_measured_values[2] - np.array(M2_theoretical_values)[2])
absolute_errors_triangle_M2 = np.abs(M2_measured_values[3] - np.array(M2_theoretical_values)[3])

# Plotting
fig, ax = plt.subplots()

# Plot points
#ax.scatter(x, M0_measured_values, color='b', label='Measured')
#ax.scatter(x, M0_theoretical_values, color='r', label='Theoretical')


# Draw lines connecting measured and theoretical points
#for i in range(len(shapes)):
    #ax.plot([x[i], x[i]], [M0_measured_values[i], M0_theoretical_values[i]], color='gray', linestyle='--')
#    ax.plot(x, M0_measured_values[i], 'bo-', label='Measured')  # 'bo-' means blue color, circle marker, and solid line
#ax.plot(x, M0_theoretical_values, 'ro-', label='Theoretical')  # 'ro-' means red color, circle marker, and solid line

ax.plot(x, absolute_errors_circle_M1, 'bo-', label='Circle')  # 'bo-' means blue color, circle marker, and solid line
ax.plot(x, absolute_errors_rectangle_M1, 'ro-', label='Rectangle')  # 'ro-' means red color, circle marker, and solid line
ax.plot(x, absolute_errors_ellipse_M1, 'go-', label='Ellipse')  # 'go-' means green color, circle marker, and solid line
ax.plot(x, absolute_errors_triangle_M1, 'yo-', label='Triangle')  # 'yo-' means yellow color, circle marker, and solid line


#ax.plot(x, relative_errors_circle_M1, 'bo-', label='Circle')  # 'bo-' means blue color, circle marker, and solid line
#ax.plot(x, relative_errors_rectangle_M1, 'ro-', label='Rectangle')  # 'ro-' means red color, circle marker, and solid line
#ax.plot(x, relative_errors_ellipse_M1, 'go-', label='Ellipse')  # 'go-' means green color, circle marker, and solid line
#ax.plot(x, relative_errors_triangle_M1, 'yo-', label='Triangle')  # 'yo-' means yellow color, circle marker, and solid line

# Annotate relative errors in between the points
#for i, (measured, theoretical, error) in enumerate(zip(M0_measured_values, M0_theoretical_values, relative_errors)):
#    mid_point = (measured + theoretical) / 2
#    ax.annotate(f'{error:.2%}', (x[i], mid_point), textcoords="offset points", xytext=(10,0), ha='left', color='black')

# Add labels, title, and legend
ax.set_xlabel('Shapes')
ax.set_ylabel('M1 Absolute error')
ax.set_yscale('log')
ax.set_xlabel('Resolution')
ax.set_title('M1 Error Comparison')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

# Show plot
plt.show()