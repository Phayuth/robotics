import numpy as np
import matplotlib.pyplot as plt

def polar2cats(r, theta, x_targ=0, y_targ=0):
    x = r*np.cos(theta) + x_targ
    y = r*np.sin(theta) + y_targ
    return x, y

def sph2cat(r, theta, phi, x_targ=0, y_targ=0, z_targ=0):
    x = r*np.sin(theta)*np.cos(phi) + x_targ
    y = r*np.sin(theta)*np.sin(phi) + y_targ
    z = r*np.cos(theta) + z_targ
    return x, y, z

def approach_circle_plt(x_targ, y_targ,radius):
    theta_coord = np.linspace(0, 2*np.pi, 90)
    x_coord, y_coord = polar2cats(radius, theta_coord, x_targ, y_targ)
    plt.plot(x_coord, y_coord)