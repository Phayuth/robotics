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

def ellipse2cats(a, b, t, x_targ=0, y_targ=0): # here t variable IS NOT theta https://en.wikipedia.org/wiki/Ellipse#Parametric_representation
    x = a*np.cos(t) + x_targ
    y = b*np.sin(t) + y_targ
    return x, y

def circle_plt(x_targ, y_targ, radius):
    theta_coord = np.linspace(0, 2*np.pi, 90)
    x_coord, y_coord = polar2cats(radius, theta_coord, x_targ, y_targ)
    plt.plot(x_coord, y_coord, "g--")

def ellipse_plt(x_targ, y_targ, a, b):
    t = np.linspace(0, 2*np.pi, 90)
    x_coord, y_coord = ellipse2cats(a, b, t, x_targ, y_targ)
    plt.plot(x_coord, y_coord, "g--")


if __name__=="__main__":

    # SECTION - circle plot
    plt.axes().set_aspect('equal')
    circle_plt(0,0,1)
    plt.show()


    # SECTION - ellipse plot
    plt.axes().set_aspect('equal')
    ellipse_plt(1,1,2,1)
    plt.show()