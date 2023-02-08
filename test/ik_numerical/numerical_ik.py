import numpy as np
import matplotlib.pyplot as plt

class robot_rr_plannar:
	def __init__(self):
		self.l1 = 1
		self.l2 = 1

	def forward_kinematic(self, theta1, theta2):
		x_1 = self.l1*np.cos(theta1) + self.l2*np.cos(theta1+theta2)
		x_2 = self.l1*np.sin(theta1) + self.l2*np.sin(theta1+theta2)
		return x_1,x_2

	def jacobian(self, theta1, theta2):
		J = np.array([[-self.l1*np.sin(theta1)-self.l2*np.sin(theta1+theta2) , -self.l2*np.sin(theta1+theta2)],
					   [self.l1*np.cos(theta1)+self.l2*np.cos(theta1+theta2) ,  self.l2*np.cos(theta1+theta2)]])
		return J

	def clampMag(self,w,d):
		if np.linalg.norm(w)<=d:
			return w
		else:
			return d*(w/np.linalg.norm(w))

	def inverse_kinematic_inverse_jac(self,theta,x_desired):
		theta_history = theta
		forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
		e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
		max_iter = 100 # for when it can't reach desired pose
		i = 0

		while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

			forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
			e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
			Jac = self.jacobian(theta[0,0],theta[1,0])
			theta = theta + np.linalg.inv(Jac).dot(e)
			print(theta)
			theta_history = np.append(theta_history, theta, axis=1)
			i+=1

		return theta,theta_history

	def inverse_kinematic_inverse_jac_ClampMag(self,theta,x_desired,Dmax):
		theta_history = theta
		forwardk =self.forward_kinematic(theta[0,0],theta[1,0])
		e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
		e = self.clampMag(e,Dmax)
		max_iter = 100 # for when it can't reach desired pose
		i = 0

		while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

			forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
			e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
			e = self.clampMag(e,Dmax)
			Jac = self.jacobian(theta[0,0],theta[1,0])
			theta = theta + np.linalg.inv(Jac).dot(e)
			print(theta)
			theta_history = np.append(theta_history, theta, axis=1)
			i+=1
		return theta,theta_history

	def cal_alpha(self,e,Jac):
		return (np.dot(np.transpose(e),Jac@np.transpose(Jac)@e))/(np.dot(np.transpose(Jac@np.transpose(Jac)@e),Jac@np.transpose(Jac)@e))
	
	def inverse_kinematic_transpose_jac(self,theta,x_desired):
		theta_history = theta
		forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
		e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
		max_iter = 100 # for when it can't reach desired pose
		i = 0

		while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

			forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
			e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
			Jac = self.jacobian(theta[0,0],theta[1,0])
			alpha = self.cal_alpha(e,Jac)
			theta = theta + alpha*np.transpose(Jac).dot(e)
			print(theta)
			theta_history = np.append(theta_history, theta, axis=1)
			i+=1
		return theta,theta_history

	def inverse_kinematic_pseudoinverse_jac(self,theta,x_desired):
		theta_history = theta
		forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
		e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
		max_iter = 100 # for when it can't reach desired pose
		i = 0

		while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

			forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
			e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
			Jac = self.jacobian(theta[0,0],theta[1,0])
			theta = theta + np.linalg.pinv(Jac).dot(e)
			print(theta)
			theta_history = np.append(theta_history, theta, axis=1)
			i+=1
		return theta,theta_history


	def inverse_kinematic_damped_least_square(self,theta,x_desired,damp_cte):
		theta_history = theta
		forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
		e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
		max_iter = 100 # for when it can't reach desired pose
		i = 0

		while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

			forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
			e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
			Jac = self.jacobian(theta[0,0],theta[1,0])
			delta_theta = np.transpose(Jac) @ np.linalg.inv(Jac@np.transpose(Jac) + np.identity(2)*(damp_cte**2)) @ e
			theta = theta + delta_theta
			print(theta)
			theta_history = np.append(theta_history, theta, axis=1)
			i+=1
		return theta,theta_history

	def inverse_kinematic_damped_least_square_ClampMag(self,theta,x_desired,damp_cte,Dmax):
		theta_history = theta
		forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
		e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
		e = self.clampMag(e,Dmax)
		max_iter = 100 # for when it can't reach desired pose
		i = 0

		while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

			forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
			e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
			e = self.clampMag(e,Dmax)
			Jac = self.jacobian(theta[0,0],theta[1,0])
			delta_theta = np.transpose(Jac) @ np.linalg.inv(Jac@np.transpose(Jac) + np.identity(2)*(damp_cte**2)) @ e
			theta = theta + delta_theta
			print(theta)
			theta_history = np.append(theta_history, theta, axis=1)
			i+=1
		return theta,theta_history

	def clampMagAbs(self,w,d): 
		if np.max(abs(w))<=d:
			return w
		else:
			return d*(w/(np.max(abs(w))))
	
	def rho11(self,theta1,theta2):
		eq1 = -self.l1*np.sin(theta1)
		eq2 =  self.l1*np.cos(theta1)
		vector = np.array([[eq1],[eq2]])
		return np.linalg.norm(vector)

	def rho12(self,theta1,theta2):
		eq1 = 0
		eq2 = 0
		vector = np.array([[eq1],[eq2]])
		return np.linalg.norm(vector)

	def rho21(self,theta1,theta2):
		eq1 = -self.l1*np.sin(theta1) - self.l2*np.sin(theta1+theta2)
		eq2 =  self.l1*np.cos(theta1) + self.l2*np.cos(theta1+theta2)
		vector = np.array([[eq1],[eq2]])
		return np.linalg.norm(vector)

	def rho22(self,theta1,theta2):
		eq1 = -self.l2*np.sin(theta1+theta2)
		eq2 =  self.l2*np.cos(theta1+theta2)
		vector = np.array([[eq1],[eq2]])
		return np.linalg.norm(vector)


	def inverse_kinematic_selectively_damped_least_square_pre(self,theta,x_desired):
		theta_history = theta
		forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
	
		e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
		
		max_iter = 100 # for when it can't reach desired pose
		i = 0

		while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

			forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
			e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
			
			Jac = self.jacobian(theta[0,0],theta[1,0])
			U, D, VT = np.linalg.svd(Jac)
			V = np.transpose(VT)
			v1 = np.array([[V[0,0]],[V[1,0]]])
			v2 = np.array([[V[0,1]],[V[1,1]]])
			u1 = np.array([[U[0,0]],[U[1,0]]])
			u2 = np.array([[U[0,1]],[U[1,1]]])
			alpha1 = np.dot(np.transpose(e),u1)
			alpha2 = np.dot(np.transpose(e),u2)

			tau1 = 0.2
			tau2 = 0.2

			delta_theta=alpha1*tau1*v1 + alpha2*tau2*v2

			theta = theta + delta_theta
			print(theta)
			theta_history = np.append(theta_history, theta, axis=1)
			i+=1
		return theta,theta_history

	def inverse_kinematic_selectively_damped_least_square(self,theta,x_desired):
		theta_history = theta
		forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
		
		e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
		
		max_iter = 100 # for when it can't reach desired pose
		i = 0

		while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

			forwardk = self.forward_kinematic(theta[0,0],theta[1,0])
			e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
			
			Jac = self.jacobian(theta[0,0],theta[1,0])
			U, D, VT = np.linalg.svd(Jac)
			V = np.transpose(VT)
			v1 = np.array([[V[0,0]],[V[1,0]]])
			v2 = np.array([[V[0,1]],[V[1,1]]])

			u1 = np.array([[U[0,0]],[U[1,0]]])
			u2 = np.array([[U[0,1]],[U[1,1]]])

			alpha1 = np.dot(np.transpose(e),u1)
			alpha2 = np.dot(np.transpose(e),u2)

			v11 = V[0,0]
			v12 = V[0,1]
			v21 = V[1,0]
			v22 = V[1,1]

			sigma1 = D[0]
			sigma2 = D[1]

			M11 = (1/sigma1)*(np.abs(v11)*self.rho11(theta[0,0],theta[1,0]) + np.abs(v21)*self.rho12(theta[0,0],theta[1,0]))
			M12 = (1/sigma1)*(np.abs(v11)*self.rho21(theta[0,0],theta[1,0]) + np.abs(v21)*self.rho22(theta[0,0],theta[1,0]))

			M21 = (1/sigma2)*(np.abs(v12)*self.rho11(theta[0,0],theta[1,0]) + np.abs(v22)*self.rho12(theta[0,0],theta[1,0]))
			M22 = (1/sigma2)*(np.abs(v12)*self.rho21(theta[0,0],theta[1,0]) + np.abs(v22)*self.rho22(theta[0,0],theta[1,0]))

			M1 = M11 + M12
			M2 = M21 + M22

			N1 = np.sum(np.abs(u1))
			N2 = np.sum(np.abs(u2))

			gama_max = np.pi/4 # user tuning
			gama1 = np.minimum(1,N1/M1)*gama_max
			gama2 = np.minimum(1,N2/M2)*gama_max

			c1 = (1/sigma1) * alpha1 * v1
			c2 = (1/sigma2) * alpha2 * v2

			phi1 = self.clampMagAbs(c1,gama1)
			phi2 = self.clampMagAbs(c2,gama2)

			delta_theta = self.clampMagAbs(phi1+phi2,gama_max)

			theta = theta + delta_theta
			print(theta)
			theta_history = np.append(theta_history, theta, axis=1)
			i+=1
		return theta,theta_history

	def plot_arm(self, theta1, theta2, *args, **kwargs):
		shoulder = np.array([0, 0])
		elbow = shoulder + np.array([self.l1 * np.cos(theta1), self.l1 * np.sin(theta1)])
		wrist = elbow + np.array([self.l2 * np.cos(theta1 + theta2), self.l2 * np.sin(theta1 + theta2)])

		plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
		plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

		plt.plot(shoulder[0], shoulder[1], 'ro')
		plt.plot(elbow[0], elbow[1], 'ro')
		plt.plot(wrist[0], wrist[1], 'ro')
		
		
		title = kwargs.get('title', None)
		plt.annotate("X pos = "+str(wrist[0]), xy=(0, 1.8+2), xycoords="data",va="center", ha="center")
		plt.annotate("Y pos = "+str(wrist[1]), xy=(0, 1.5+2), xycoords="data",va="center", ha="center")
		
		circle1 = plt.Circle((0, 0), self.l1+self.l2,alpha=0.5, edgecolor='none')
		plt.gca().add_patch(circle1)
		plt.title(title)
		
		plt.xlim(-3, 3)
		plt.ylim(-2, 4)

		plt.show()