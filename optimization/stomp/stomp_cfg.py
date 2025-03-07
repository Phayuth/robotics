group_name = "panda_arm"
optimization = None
num_timesteps = 60
num_iterations = 40
num_iterations_after_valid = 0
num_rollouts = 30
max_rollouts = 30
initialization_method = 1  # [1 = LINEAR_INTERPOLATION, 2 = CUBIC_POLYNOMIAL, 3 = MININUM_CONTROL_COST]
control_cost_weight = 0.0
task = None


class noise_generator:
    name = "stomp_moveit/NormalDistributionSampling"
    stddev = [0.05, 0.8, 1.0, 0.8, 0.4, 0.4, 0.4]


class cost_functions:
    name = "stomp_moveit/CollisionCheck"
    collision_penalty = 1.0
    cost_weight = 1.0
    kernel_window_percentage = 0.2
    longest_valid_joint_move = 0.05


class noisy_filters:
    name = "stomp_moveit/JointLimits"
    lock_start = True
    lock_goal = True
    name = "stomp_moveit/MultiTrajectoryVisualization"
    line_width = 0.02
    rgb = [255, 255, 0]
    marker_array_topic = "stomp_trajectories"
    marker_namespace = "noisy"


class update_filters:
    name = "stomp_moveit/PolynomialSmoother"
    poly_order = 6
    name = "stomp_moveit/TrajectoryVisualization"
    line_width = 0.05
    rgb = [0, 191, 255]
    error_rgb = [255, 0, 0]
    publish_intermediate = True
    marker_topic = "stomp_trajectory"
    marker_namespace = "optimized"


# https://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/stomp_planner/stomp_planner_tutorial.html
