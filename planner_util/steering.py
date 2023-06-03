import numpy as np

def steer(base_config, random_config, eta):
    direction = np.array(random_config) - np.array(base_config)
    distance = np.linalg.norm(direction)
    
    if distance <= eta:
        # If the distance to the random configuration is less than or equal to eta,
        # return the random configuration itself as the new node.
        new_config = random_config
    else:
        # Normalize the direction vector and scale it by the step size (eta) to get
        # the displacement vector from the base configuration.
        displacement = (direction / distance) * eta
        
        # Add the displacement vector to the base configuration to get the new node.
        new_config = tuple(np.array(base_config) + displacement)
    
    return new_config

base_config = (0, 0, 0, 0, 0, 0)  # Base configuration
random_config = (5, 3, 3, 3, 3, 3)  # Random configuration
eta = 1.6  # Step size

new_config = steer(base_config, random_config, eta)
print(new_config)

norm = np.linalg.norm(new_config)
print(norm)


import matplotlib.pyplot as plt

def interpolate(start, end, num_segments):
    segment_points = []
    step_size = [(end[i] - start[i]) / num_segments for i in range(len(start))]
    
    for segment in range(num_segments + 1):
        point = [start[i] + step_size[i] * segment for i in range(len(start))]
        segment_points.append(point)
    
    return segment_points

start = [0, 0, 0, 0, 0, 0]
end = [10, 10, 10, 10, 10, 10]
num_segments = 5

segment_points = interpolate(start, end, num_segments)

# Extract x, y, z, w, v, u coordinates from segment points
x_coords = [point[0] for point in segment_points]
y_coords = [point[1] for point in segment_points]
z_coords = [point[2] for point in segment_points]
w_coords = [point[3] for point in segment_points]
v_coords = [point[4] for point in segment_points]
u_coords = [point[5] for point in segment_points]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_coords, y_coords, z_coords, marker='o', label='Segment Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title('Segment Points on the Line in 6D')
plt.legend()
plt.show()