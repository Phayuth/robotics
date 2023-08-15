import numpy as np
import matplotlib.pyplot as plt

# Define the range of joint angles (e.g., from 0 to 360 degrees)
joint_min = 0
joint_max = 360

# Define the start and goal configurations
start = np.array([30, 45])  # [Joint1 angle, Joint2 angle]
goal = np.array([300, 200])  # [Joint1 angle, Joint2 angle]

# Function to calculate the toroidal distance between two joint angles
def toroidal_distance(angle1, angle2):
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, joint_max - diff)

# Function to steer from a source configuration to a target configuration
def steer(source, target, max_step):
    diff = np.linalg.norm(target - source)
    if diff > max_step:
        diff -= joint_max
    elif diff < -max_step:
        diff += joint_max
    return source + diff

def is_collision_free(config):
    return True

# RRT Planning function
def rrt_planning(start, goal, num_iterations, max_step_size):
    tree = [start]

    for _ in range(num_iterations):
        # Sample a random configuration
        random_config = np.random.uniform(joint_min, joint_max, 2)

        # Find the nearest neighbor in the tree
        nearest_neighbor = min(tree, key=lambda node: np.linalg.norm(node - random_config))

        # Steer towards the random configuration
        new_config = steer(nearest_neighbor, random_config, max_step_size)

        # Append the new configuration to the tree if collision-free
        if is_collision_free(new_config):  # You'll need to implement this function
            tree.append(new_config)

    return tree

# Function to visualize the RRT tree and path
def visualize_rrt(tree, path=[]):
    plt.figure(figsize=(8, 8))
    plt.plot([node[0] for node in tree], [node[1] for node in tree], 'go', markersize=4)

    for i in range(len(tree) - 1):
        plt.plot([tree[i][0], tree[i + 1][0]], [tree[i][1], tree[i + 1][1]], 'b', linewidth=1)

    if path:
        plt.plot([node[0] for node in path], [node[1] for node in path], 'r', linewidth=2)

    plt.plot(start[0], start[1], 'ro', markersize=6)
    plt.plot(goal[0], goal[1], 'ro', markersize=6)

    plt.xlim(joint_min, joint_max)
    plt.ylim(joint_min, joint_max)
    plt.xlabel('Joint 1 Angle')
    plt.ylabel('Joint 2 Angle')
    plt.title('RRT Planning for 2-Joint Robot Arm')
    plt.grid(True)
    plt.show()

# Example usage
num_iterations = 1000
max_step_size = 45
tree = rrt_planning(start, goal, num_iterations, max_step_size)

# Extract the path from the tree (simplest approach: just use the goal configuration)
path = [goal]

# Visualize the RRT tree and path
visualize_rrt(tree, path)
