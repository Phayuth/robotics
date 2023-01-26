import numerical_ik
import numpy as np

robot = numerical_ik.robot_rr_plannar()

theta = np.array([[0.1],[0.1]]) # vector 2x1
x_desired = np.array([[0.5],[0.5]]) # vector 2x1

# x_1 , x_2 = robot.forward_kinematic(theta[0,0],theta[1,0])
# theta,theta_history = robot.inverse_kinematic_inverse_jac(theta,x_desired)
# theta,theta_history = robot.inverse_kinematic_damped_least_square_ClampMag(theta,x_desired,0.5,0.5)
# theta,theta_history = robot.inverse_kinematic_transpose_jac(theta,x_desired)
# robot.plot_arm(theta[0,0],theta[1,0],title='The edge of the circle is singularity')

# theta,theta_history = robot.inverse_kinematic_selectively_damped_least_square_pre(theta,x_desired)
theta,theta_history = robot.inverse_kinematic_selectively_damped_least_square(theta,x_desired)
robot.plot_arm(theta[0,0],theta[1,0],title='The edge of the circle is singularity')