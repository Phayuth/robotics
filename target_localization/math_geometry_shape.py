import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, sign


def superellipse(theta):
    a = 10  # tune parameter
    b = 10  # tune parameter
    n = 4  # tune parameter

    x = ((abs(cos(theta)))**(2 / n)) * (a * sign(cos(theta)))
    y = ((abs(sin(theta)))**(2 / n)) * (b * sign(sin(theta)))

    return x, y


def superellipsoids_3d(eta, omega):
    a1 = 1
    a2 = 1
    a3 = 1
    ep1 = 0.1
    ep2 = 0.1

    #check if ep1 or ep2 is smaller than 1, if yes then cos and sin must be sign(cos) and sign(sin)
    # if ep1 < 1 or ep2 < 1:
    #     x = a1*(sign(cos(eta))**ep1)*(sign(cos(omega))**ep2)
    #     y = a2*(sign(cos(eta))**ep1)*(sign(sin(omega))**ep2)
    #     z = a3*(sign(sin(eta))**ep1)

    # else:
    x = a1 * (cos(eta)**ep1) * (cos(omega)**ep2)
    y = a2 * (cos(eta)**ep1) * (sin(omega)**ep2)
    z = a3 * (sin(eta)**ep1)

    return [x, y, z]


def surface_normal_supelp(eta, omega):
    a1 = 1
    a2 = 1
    a3 = 1
    ep1 = 1
    ep2 = 1

    x = (1/a1) * (np.cos(eta)**(2 - ep1)) * (np.cos(omega)**(2 - ep2))
    y = (1/a2) * (np.cos(eta)**(2 - ep2)) * (np.sin(omega)**(2 - ep2))
    z = (1/a3) * (np.sin(eta)**(2 - ep1))

    return [x, y, z]


if __name__ == "__main__":
    # super ellipsoid 2d
    theta = np.linspace(0, 2 * np.pi, 360)
    x, y = superellipse(theta)

    plt.plot(x, y)
    plt.show()

    # super ellipsoid 3d
    eta = np.linspace(0, np.pi, 25)
    omega = np.linspace(0, 2 * np.pi, 25)
    ax = plt.axes(projection='3d')
    for et in eta:
        for om in omega:
            point = superellipsoids_3d(et, om)
            ax.scatter(point[0], point[1], point[2])

    plt.show()