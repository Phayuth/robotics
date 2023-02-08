import numpy as np
from clampMag import clampMag

# THIS METHOD OF IK IS NOT WORKING WHEN THE ARM IS AT SINGULARITY
# I MADE THIS JUST TO SHOW AND PROOF OF THEORY, DON'T USE THIS IN THE ACTUAL EXPERIMENT

class ik_jacobian_inverse:
	def __init__(self,max_iteration,robot_class):
		self.max_iter = max_iteration # for when it can't reach desired pose
		self.robot = robot_class
		self.theta_history = np.array([[]])

	def inverse_jac(self,theta_current,x_desired):
		x_current = self.robot.forward_kinematic(theta_current)
		e = x_desired - x_current
		i = 0

		while np.linalg.norm(e) > 0.001 and i < self.max_iter:      # norm = sqrt(x^2 + y^2)

			x_current = self.robot.forward_kinematic(theta_current)
			e = x_desired - x_current
			Jac = self.robot.jacobian(theta_current)
			theta_current = theta_current + np.linalg.inv(Jac).dot(e)
			i+=1

		return theta_current

	def inverse_jac_ClampMag(self,theta_current,x_desired,Dmax):
		x_current = self.robot.forward_kinematic(theta_current)
		e = x_desired - x_current
		e = clampMag(e,Dmax)
		i = 0

		while np.linalg.norm(e) > 0.001 and i < self.max_iter:

			x_current = self.robot.forward_kinematic(theta_current)
			e = x_desired - x_current
			e = clampMag(e,Dmax)
			Jac = self.robot.jacobian(theta_current)
			theta_current = theta_current + np.linalg.inv(Jac).dot(e)
			i+=1
			
		return theta_current