import numpy as np

def polar2cats(r, theta, x_targ=0, y_targ=0):
    x = r*np.cos(theta) + x_targ
    y = r*np.sin(theta) + y_targ
    return x, y

def sph2cat(r, theta, phi, x_targ=0, y_targ=0, z_targ=0):
    x = r*np.sin(theta)*np.cos(phi) + x_targ
    y = r*np.sin(theta)*np.sin(phi) + y_targ
    z = r*np.cos(theta) + z_targ
    return x, y, z

