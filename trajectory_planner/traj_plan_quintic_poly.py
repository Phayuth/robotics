import numpy as np
import matplotlib.pyplot as plt


def quintic_polynomial(theta_0,theta_0_dot,theta_0_ddot,theta_f,theta_f_dot,theta_f_ddot,t,tf):

    a0 = theta_0
    a1 = theta_0_dot
    a2 = theta_0_ddot / 2
    a3 = ((20*theta_f) - (20*theta_0) - ((12*theta_0_dot + 8*theta_f_dot)*tf) - ((3*theta_0_ddot - theta_f_ddot)*(tf**2)))/(2*(tf**3))
    a4 = (-(30*theta_f) + (30*theta_0) + ((16*theta_0_dot + 14*theta_f_dot)*tf) + ((3*theta_0_ddot - 2*theta_f_ddot)*(tf**2)))/(2*(tf**4))
    a5 = ((12*theta_f) - (12*theta_0) - ((6*theta_0_dot + 6*theta_f_dot)*tf) - ((theta_0_ddot - theta_f_ddot)*(tf**2)))/(2*(tf**5))
    
    # print(a0,a1,a2,a3,a4,a5)

    theta = a0 + (a1*t) + (a2*(t**2)) + (a3*(t**3)) + (a4*(t**4)) + (a5*(t**5))
    theta_dot = a1 + 2*a2*t + 3*a3*(t**2) + 4*a4*(t**3) + 5*a5*(t**4)
    theta_ddot = 2*a2 + 6*a3*t + 12*a4*(t**2) + 20*a5*(t**3)

    return theta,theta_dot,theta_ddot

theta_0 = 0
v_0 = 0
a_0 = 0

theta_f = 5
v_f = 0
a_f = 0
t_f = 5

t = np.arange(0,t_f,0.01)

theta,theta_dot,theta_ddot = quintic_polynomial(theta_0,v_0,a_0,theta_f,v_f,a_f,t,t_f)

# print(t,theta,theta_dot,theta_ddot)

plt.plot(t,theta)
plt.plot(t,theta_dot)
plt.plot(t,theta_ddot)
plt.show()