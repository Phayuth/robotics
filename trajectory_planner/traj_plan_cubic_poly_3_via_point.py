import numpy as np
import matplotlib.pyplot as plt


def cubic_3_via_point(theta_0,theta_v,theta_g,t,tf):
    # case tf = tf1 = tf2
    a10 = theta_0
    a11 = 0 
    a12 = (12*theta_v - 3*theta_g - 9*theta_0)/(4*(tf**2))
    a13 = (-8*theta_v + 3*theta_g + 5*theta_0)/(4*(tf**3))

    a20 = theta_v
    a21 = (3*theta_g - 3*theta_0)/(4*tf)
    a22 = (-12*theta_v + 6*theta_g + 6*theta_0)/(4*(tf**2))
    a23 = (8*theta_v - 5*theta_g - 3*theta_0)/(4*(tf**3))

    seg1_theta = a10 + (a11*t) + (a12*(t**2)) + (a13*(t**3))
    seg1_theta_dot = a11 + 2*a12*t + 3*a13*(t**2)
    seg1_theta_ddot = 2*a12 + 6*a13*t

    seg2_theta = a20 + (a21*t) + (a22*(t**2)) + (a23*(t**3))
    seg2_theta_dot = a21 + 2*a22*t + 3*a23*(t**2)
    seg2_theta_ddot = 2*a22 + 6*a23*t

    return seg1_theta,seg1_theta_dot,seg1_theta_ddot,seg2_theta,seg2_theta_dot,seg2_theta_ddot

theta_0 = 0
theta_v = 5
theta_g = 10
t_f = 10

t = np.arange(0,t_f,0.01)

seg1_theta,seg1_theta_dot,seg1_theta_ddot,seg2_theta,seg2_theta_dot,seg2_theta_ddot = cubic_3_via_point(theta_0,theta_v,theta_g,t,t_f)

plt.plot(t,seg1_theta)
plt.plot(t,seg2_theta)
plt.show()