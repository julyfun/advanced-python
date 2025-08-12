import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Set adjustable ranges
x_range = (-2, 3)
y_range = (-2, 3)
z_range = (-3, 5)

# Create meshgrid for x and y
x = np.linspace(x_range[0], x_range[1], 100)
y = np.linspace(y_range[0], y_range[1], 100)
X, Y = np.meshgrid(x, y)


# Function for z = xy
def z_func(x, y):
    return x * y


# Calculate z values for the z = xy surface
Z = z_func(X, Y)

# Plot the z = xy surface
surface = ax.plot_surface(
    X, Y, Z, cmap=cm.coolwarm, alpha=0.7, linewidth=0, label="z = xy"
)

# Create the z = 0 plane
xx, yy = np.meshgrid(x_range, y_range)
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, color="blue", alpha=0.3, label="z = 0")

# Create the x + y - 1 = 0 plane
# Rearranging to z form: z can be any value for points where x + y - 1 = 0
# We'll create a vertical plane along this line
x_plane = np.linspace(x_range[0], x_range[1], 20)
y_plane = 1 - x_plane
z_plane = np.linspace(z_range[0], z_range[1], 20)
X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
Y_plane = 1 - X_plane
ax.plot_surface(
    X_plane, Y_plane, Z_plane, color="green", alpha=0.3, label="x + y - 1 = 0"
)

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of z = 0, x + y - 1 = 0, and z = xy")

# Set axis limits
ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_zlim(z_range)

# Add a marker at the origin (0,0,0)
ax.scatter([0], [0], [0], color="red", s=100, marker="o", label="Origin (0,0,0)")

# Add coordinate axes
# X-axis
ax.plot([x_range[0], x_range[1]], [0, 0], [0, 0], "r-", linewidth=2, label="X-axis")
# Y-axis
ax.plot([0, 0], [y_range[0], y_range[1]], [0, 0], "g-", linewidth=2, label="Y-axis")
# Z-axis
ax.plot([0, 0], [0, 0], [z_range[0], z_range[1]], "b-", linewidth=2, label="Z-axis")

# Add a custom legend
from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color="blue", lw=4, alpha=0.7),
    Line2D([0], [0], color="green", lw=4, alpha=0.7),
    Line2D([0], [0], color=cm.coolwarm(0.5), lw=4),
    Line2D([0], [0], color="red", marker="o", markersize=10, linestyle="None"),
    Line2D([0], [0], color="red", lw=2),
    Line2D([0], [0], color="green", lw=2),
    Line2D([0], [0], color="blue", lw=2),
]
ax.legend(
    custom_lines,
    [
        "z = 0",
        "x + y - 1 = 0",
        "z = xy",
        "Origin (0,0,0)",
        "X-axis",
        "Y-axis",
        "Z-axis",
    ],
    loc="upper left",
)

# Calculate and plot intersection points

# 1. Intersection of z = 0 and x + y - 1 = 0 (a line in the z = 0 plane)
x_line = np.linspace(x_range[0], x_range[1], 100)
y_line = 1 - x_line
z_line = np.zeros_like(x_line)
# Only keep points within our y range
valid_points = (y_line >= y_range[0]) & (y_line <= y_range[1])
ax.plot(
    x_line[valid_points],
    y_line[valid_points],
    z_line[valid_points],
    "purple",
    linewidth=3,
    label="z=0 ∩ (x+y-1=0)",
)

# 2. Intersection of z = xy and x + y - 1 = 0
# When we have points on the plane x + y - 1 = 0, then y = 1 - x
# Substituting into z = xy: z = x(1-x) = x - x²
x_curve = np.linspace(x_range[0], x_range[1], 100)
y_curve = 1 - x_curve
z_curve = x_curve * y_curve  # or x_curve - x_curve**2
# Only keep points within our ranges
valid_points = (y_curve >= y_range[0]) & (y_curve <= y_range[1])
ax.plot(
    x_curve[valid_points],
    y_curve[valid_points],
    z_curve[valid_points],
    "orange",
    linewidth=3,
    label="z=xy ∩ (x+y-1=0)",
)

# 3. Intersection of z = xy and z = 0 (when xy = 0, means either x=0 or y=0)
# Path along x=0, y varies
x_path1 = np.zeros(100)
y_path1 = np.linspace(y_range[0], y_range[1], 100)
z_path1 = np.zeros(100)  # z = x*y = 0*y = 0
ax.plot(x_path1, y_path1, z_path1, "cyan", linewidth=3, label="z=xy ∩ z=0 (x=0)")

# Path along y=0, x varies
x_path2 = np.linspace(x_range[0], x_range[1], 100)
y_path2 = np.zeros(100)
z_path2 = np.zeros(100)  # z = x*y = x*0 = 0
ax.plot(x_path2, y_path2, z_path2, "magenta", linewidth=3, label="z=xy ∩ z=0 (y=0)")

# 4. Triple intersection point (where all three surfaces meet)
# Solve: z = 0, x + y - 1 = 0, z = xy
# From z = 0 and z = xy, we have xy = 0, so either x = 0 or y = 0
# If x = 0, then from x + y - 1 = 0, we get y = 1
# If y = 0, then from x + y - 1 = 0, we get x = 1
triple_intersect_points = [(0, 1, 0), (1, 0, 0)]
triple_x, triple_y, triple_z = zip(*triple_intersect_points)
ax.scatter(
    triple_x,
    triple_y,
    triple_z,
    color="yellow",
    s=150,
    edgecolor="black",
    label="Triple Intersection",
)

# Update the legend
custom_lines.extend(
    [
        Line2D([0], [0], color="purple", lw=3),
        Line2D([0], [0], color="orange", lw=3),
        Line2D([0], [0], color="cyan", lw=3),
        Line2D([0], [0], color="magenta", lw=3),
        Line2D(
            [0],
            [0],
            color="yellow",
            marker="o",
            markersize=10,
            markeredgecolor="black",
            linestyle="None",
        ),
    ]
)

ax.legend(
    custom_lines,
    [
        "z = 0",
        "x + y - 1 = 0",
        "z = xy",
        "Origin (0,0,0)",
        "X-axis",
        "Y-axis",
        "Z-axis",
        "z=0 ∩ (x+y-1=0)",
        "z=xy ∩ (x+y-1=0)",
        "z=xy ∩ z=0 (x=0)",
        "z=xy ∩ z=0 (y=0)",
        "Triple Intersection",
    ],
    loc="upper left",
    fontsize=8,
)

# Show the plot
plt.tight_layout()
plt.show()
0
