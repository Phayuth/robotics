import numpy as np
import matplotlib.pyplot as plt


def cubic_via_point(theta_0,theta_0_dot,theta_0_ddot,theta_f,theta_f_dot,theta_f_ddot,t,tf):

    a0 = theta_0
    a1 = theta_0_dot
    a2 = (3*(theta_f - theta_0))/(tf**2) - (2*theta_0_dot)/(tf) - (theta_f_dot)/(tf)
    a3 = (-2*(theta_f - theta_0))/(tf**3) + (theta_f_dot + theta_0_dot)/(tf**2)
    
    theta = a0 + (a1*t) + (a2*(t**2)) + (a3*(t**3))
    theta_dot = a1 + 2*a2*t + 3*a3*(t**2)
    theta_ddot = 2*a2 + 6*a3*t

    return theta,theta_dot,theta_ddot

theta_0 = 0
v_0 = 0
a_0 = 0

theta_f = 5
v_f = 0
a_f = 0
t_f = 5

t = np.arange(0,t_f,0.01)

theta,theta_dot,theta_ddot = cubic_via_point(theta_0,v_0,a_0,theta_f,v_f,a_f,t,t_f)

# print(t,theta,theta_dot,theta_ddot)

plt.plot(t,theta)
plt.plot(t,theta_dot)
plt.plot(t,theta_ddot)
plt.show()