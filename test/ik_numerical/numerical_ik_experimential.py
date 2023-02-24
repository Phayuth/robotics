# Library
import numpy as np
import matplotlib.pyplot as plt

# Function
# <img src="attachment:edf15534-c2fb-487e-8c58-8d68e7f14776.png" alt="Drawing" style="width: 500px;"/></td>

def plot_arm(theta1, theta2, *args, **kwargs):
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    wrist = elbow + np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])

    plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
    plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

    plt.plot(shoulder[0], shoulder[1], 'ro')
    plt.plot(elbow[0], elbow[1], 'ro')
    plt.plot(wrist[0], wrist[1], 'ro')
    
    
    title = kwargs.get('title', None)
    plt.annotate("X pos = "+str(wrist[0]), xy=(0, 1.8+2), xycoords="data",va="center", ha="center")
    plt.annotate("Y pos = "+str(wrist[1]), xy=(0, 1.5+2), xycoords="data",va="center", ha="center")
    
    circle1 = plt.Circle((0, 0), l1+l2,alpha=0.5, edgecolor='none')
    plt.gca().add_patch(circle1)
    plt.title(title)
    
    plt.xlim(-3, 3)
    plt.ylim(-2, 4)

    plt.show()

# Forward Kinematic
# x_1 is endeffector pose x , x_2 is endeffector pose y
def fk(theta1,theta2):
    x_1 = l1*np.cos(theta1) + l2*np.cos(theta1+theta2)
    x_2 = l1*np.sin(theta1) + l2*np.sin(theta1+theta2)
    return x_1,x_2

# check if the matrix is square
def is_square(J):
    # if the row and column is equal each other that the mat is square
    # J[1,:] = len of column
    # J[:,1] = len of row
    if len(J[1,:]) == len(J[:,1]):
        return True
    else:
        return False

# calcualte Velocity at the tip
def vtip(J1,J2,theta_dot1,theta_dot2):
    return J1*theta_dot1 + J2*theta_dot2

# Jacobian
def jacobian_mat(theta1,theta2):
    J = np.array([[-l1*np.sin(theta1)-l2*np.sin(theta1+theta2) , -l2*np.sin(theta1+theta2)],
                   [l1*np.cos(theta1)+l2*np.cos(theta1+theta2) , l2*np.cos(theta1+theta2)]])
    return J

def plot_arm_history(theta_history):
    for i in range(np.shape(theta_history)[1]-1):
        plot_arm(theta_history[0,i],theta_history[1,i])

# Input
l1 = 1
l2 = 1
theta1 = np.pi/4
theta2 = np.pi/4
x_1 , x_2 = fk(theta1,theta2)
print("x_1 is : %f, x_2 is : %f"%(x_1,x_2))

plot_arm(theta1,theta2,title='The edge of the circle is singularity')

# <img src="attachment:8db1a4e2-156f-41ce-a7a6-8cecb1b2690c.png" alt="Drawing" style="width: 500px;"/></td>

J = jacobian_mat(theta1,theta2)

# Jacobian Each theta
J1 = np.array([[-l1*np.sin(theta1)-l2*np.sin(theta1+theta2)],
               [l1*np.cos(theta1)+l2*np.cos(theta1+theta2)]])

J2 = np.array([[-l2*np.sin(theta1+theta2)],
               [l2*np.cos(theta1+theta2)]])

print(J1)
print(J2)

is_square(J)

# Determine the J determenant
np.linalg.det(J)
# the det(J) is not 0 , and J is square -> invertible
# the det(J) is 0 or J is square -> uninvertible -> singular matrix ( even the numpy, tell you it is singular matrix)

inv_J = np.linalg.inv(J) # simple inverse
pinv_J = np.linalg.pinv(J) # using pseudo inverse
print(inv_J)
print(pinv_J)

# calcualte vtip
v_tip = vtip(J1,J2,0.2,0.2)
print(v_tip)

# Normal Inverse Jacobian IK Newton Raspson

# <img src="attachment:f801befd-969c-46d4-89f0-fda8be8c35af.png" alt="Drawing" style="width: 500px;"/>

def ik_jac_normal_inv(theta,x_desired):
    theta_history = theta
    forwardk = fk(theta[0,0],theta[1,0])
    e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
    max_iter = 100 # for when it can't reach desired pose
    i = 0

    while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

        forwardk = fk(theta[0,0],theta[1,0])
        e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
        Jac = jacobian_mat(theta[0,0],theta[1,0])
        theta = theta + np.linalg.inv(Jac).dot(e)
        print(theta)
        theta_history = np.append(theta_history, theta, axis=1)
        i+=1
    return theta,theta_history

# Setting target positions closer (with Clamp Magnitude)

def clampMag(w,d):
    if np.linalg.norm(w)<=d:
        return w
    else:
        return d*(w/np.linalg.norm(w))

# test
t = np.array([[0.5],[0.5]])
Dmax = 0.5
e = clampMag(t,Dmax)
print(e)
# https://stackoverflow.com/questions/55502515/how-to-clamp-a-vector-within-some-magnitude
# ([0.5, 0.5], 1) = [0.5, 0.5] answer
# ([0.5, 0.5], 0.5) = [0.35355339059327373, 0.35355339059327373] answer

def ik_jac_clampMag(theta,x_desired,Dmax):
    theta_history = theta
    forwardk = fk(theta[0,0],theta[1,0])
    e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
    e = clampMag(e,Dmax)
    max_iter = 100 # for when it can't reach desired pose
    i = 0

    while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

        forwardk = fk(theta[0,0],theta[1,0])
        e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
        e = clampMag(e,Dmax)
        Jac = jacobian_mat(theta[0,0],theta[1,0])
        theta = theta + np.linalg.inv(Jac).dot(e)
        print(theta)
        theta_history = np.append(theta_history, theta, axis=1)
        i+=1
    return theta,theta_history

# Jacobian Transpose Method

def cal_alpha(e,Jac):
    return (np.dot(np.transpose(e),Jac@np.transpose(Jac)@e))/(np.dot(np.transpose(Jac@np.transpose(Jac)@e),Jac@np.transpose(Jac)@e))

def ik_jac_transpose(theta,x_desired):
    theta_history = theta
    forwardk = fk(theta[0,0],theta[1,0])
    e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
    max_iter = 100 # for when it can't reach desired pose
    i = 0

    while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

        forwardk = fk(theta[0,0],theta[1,0])
        e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
        Jac = jacobian_mat(theta[0,0],theta[1,0])
        alpha = cal_alpha(e,Jac)
        theta = theta + alpha*np.transpose(Jac).dot(e)
        print(theta)
        theta_history = np.append(theta_history, theta, axis=1)
        i+=1
    return theta,theta_history

# Pseudoinverse Method

def ik_jac_pseudo_inv(theta,x_desired):
    theta_history = theta
    forwardk = fk(theta[0,0],theta[1,0])
    e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
    max_iter = 100 # for when it can't reach desired pose
    i = 0

    while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

        forwardk = fk(theta[0,0],theta[1,0])
        e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
        Jac = jacobian_mat(theta[0,0],theta[1,0])
        theta = theta + np.linalg.pinv(Jac).dot(e)
        print(theta)
        theta_history = np.append(theta_history, theta, axis=1)
        i+=1
    return theta,theta_history

# Damped Least Square

def ik_damped_LS(theta,x_desired,damp_cte):
    theta_history = theta
    forwardk = fk(theta[0,0],theta[1,0])
    e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
    max_iter = 100 # for when it can't reach desired pose
    i = 0

    while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

        forwardk = fk(theta[0,0],theta[1,0])
        e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
        Jac = jacobian_mat(theta[0,0],theta[1,0])
        delta_theta = np.transpose(Jac) @ np.linalg.inv(Jac@np.transpose(Jac) + np.identity(2)*(damp_cte**2)) @ e
        theta = theta + delta_theta
        print(theta)
        theta_history = np.append(theta_history, theta, axis=1)
        i+=1
    return theta,theta_history

# Damped Least Square Prime (DLS with ClampMag)

def ik_damped_LSP(theta,x_desired,damp_cte,Dmax):
    theta_history = theta
    forwardk = fk(theta[0,0],theta[1,0])
    e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
    e = clampMag(e,Dmax)
    max_iter = 100 # for when it can't reach desired pose
    i = 0

    while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

        forwardk = fk(theta[0,0],theta[1,0])
        e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
        e = clampMag(e,Dmax)
        Jac = jacobian_mat(theta[0,0],theta[1,0])
        delta_theta = np.transpose(Jac) @ np.linalg.inv(Jac@np.transpose(Jac) + np.identity(2)*(damp_cte**2)) @ e
        theta = theta + delta_theta
        print(theta)
        theta_history = np.append(theta_history, theta, axis=1)
        i+=1
    return theta,theta_history

# Selectivly Damped Least Square

# ClampMaxAbs(w, d) is deﬁned just like ClampMag, but using the 1-norm instead of the Euclidean norm.
# (the 1-norm of w is the maximum of the absolute values of the components of w)
def clampMagAbs(w,d): 
    if abs(w).max()<=d:
        return w
    else:
        return d*(w/abs(w).max())

def ik_SDLS(theta,x_desired):
    theta_history = theta
    forwardk = fk(theta[0,0],theta[1,0])
    
    gamma_max = np.pi/4 # user tuning
    
    e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
    e = clampMag(e,Dmax)
    max_iter = 100 # for when it can't reach desired pose
    i = 0

    while np.linalg.norm(e) > 0.001 and i < max_iter:      # norm = sqrt(x^2 + y^2)

        forwardk = fk(theta[0,0],theta[1,0])
        e = x_desired - np.array([[forwardk[0]],[forwardk[1]]])
        e = clampMag(e,Dmax)
        Jac = jacobian_mat(theta[0,0],theta[1,0])
        delta_theta = np.transpose(Jac) @ np.linalg.inv(Jac@np.transpose(Jac) + np.identity(2)*(damp_cte**2)) @ e
        theta = theta + delta_theta
        print(theta)
        theta_history = np.append(theta_history, theta, axis=1)
        i+=1
    return theta,theta_history

# Apply

## Case Desired pose is in the task space 

theta = np.array([[0],[0]]) # vector 2x1
x_desired = np.array([[1],[0.5]]) # vector 2x1

## Case Desired pose is outside of task space

theta = np.array([[0],[0]]) # vector 2x1
x_desired = np.array([[0],[2.5]]) # vector 2x1

## Calculate and Plot

# theta,theta_history = ik_damped_LS(theta,x_desired,0.5)        # damped Least Square
# theta,theta_history = ik_damped_LSP(theta,x_desired,0.5,0.5)   # damped Least Square with clampMag
theta,theta_history = ik_jac_transpose(theta,x_desired)          # Jac transpose

# plot_arm(theta[0,0],theta[1,0],title="Case of Desired Pose is inside of task space(1,0.5)")
plot_arm(theta[0,0],theta[1,0],title="Case of Desired Pose is outside of task space (0,2.5)")

#### Testing Small Function

abs(np.array([[2],[-6]])).max()

clampMagAbs(np.array([[2],[-6]]),5)

p = np.array([[2],[2]])
y = np.array([[2],[2]])

J = np.ones((2,2))

print(p)
print(y)

print(J)

(np.dot(p,J@np.transpose(J)@p))/(np.dot(J@np.transpose(J)@p,J@np.transpose(J)@p))

np.dot(np.transpose(p),y)


