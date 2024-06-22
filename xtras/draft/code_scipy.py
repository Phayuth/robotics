import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
import scipy.interpolate


def linear_interp():
    x = np.linspace(0, 10, num=11)
    y = np.cos(-x**2 / 9.0)
    xnew = np.linspace(0, 10, num=1001)
    ynew = np.interp(xnew, x, y)
    plt.plot(xnew, ynew, '-', label='linear interp')
    plt.plot(x, y, 'o', label='data')
    plt.legend(loc='best')
    plt.show()


def cubic_spline_interp():
    x = np.linspace(0, 10, num=11)
    y = np.cos(-x**2 / 9.)
    spl = scipy.interpolate.CubicSpline(x, y)
    spld = spl.derivative(nu=1)
    spldd = spld.derivative(nu=1)
    splddd = spldd.derivative(nu=1)

    xnew = np.linspace(0, 10, num=1001)
    ynew = spl(xnew)  # evalute
    ydnew = spld(xnew)
    yddnew = spldd(xnew)
    ydddnew = splddd(xnew)

    fig, ax = plt.subplots(4, 1, figsize=(5, 7))
    ax[0].plot(x, y, 'o', label='data')
    ax[0].plot(xnew, ynew)
    ax[1].plot(xnew, ydnew, '--', label='1st derivative')
    ax[2].plot(xnew, yddnew, '--', label='2nd derivative')
    ax[3].plot(xnew, ydddnew, '--', label='3rd derivative')
    plt.tight_layout()
    plt.show()


def monotone_interp():  # prevent overshooting
    x = np.array([1., 2., 3., 4., 4.5, 5., 6., 7., 8])
    y = x**2
    y[4] += 101
    xx = np.linspace(1, 8, 101)
    plt.plot(xx, scipy.interpolate.CubicSpline(x, y)(xx), '--', label='spline')
    plt.plot(xx, scipy.interpolate.Akima1DInterpolator(x, y)(xx), '-', label='Akima1D')
    plt.plot(xx, scipy.interpolate.PchipInterpolator(x, y)(xx), '-', label='pchip')
    plt.plot(x, y, 'o')
    plt.legend()
    plt.show()


def bspline_interp():
    x = np.linspace(0, 100, 10)
    y = np.sin(x)
    bspl = scipy.interpolate.make_interp_spline(x, y, k=3)  #k is degree, k=3 is cubic
    xx = np.linspace(0, 100, 100)
    ynew = bspl(xx)
    ydnew = bspl.derivative()(xx)  # a BSpline representing the derivative
    yddnew = bspl.derivative().derivative()(xx)
    plt.plot(x, y, 'o', label='data')
    plt.plot(xx, ynew, '--', label='aprox')
    plt.plot(xx, ydnew, '--', label='aprox derivative')
    # plt.plot(xx, yddnew, '--', label='$d \sin(\pi x)/dx / \pi$ approx')
    plt.legend()
    plt.show()


def smoothing_spline_interp():
    x = np.arange(0, 2 * np.pi + np.pi / 4, 2 * np.pi / 16)
    y = np.sin(x) + 0.4 * np.random.default_rng().standard_normal(size=len(x))

    tck = scipy.interpolate.splrep(x, y, s=0)  # The normal output is a 3-tuple, containing the knot-points=t, the coefficients=c and the order of the spline=k.
    tck_s = scipy.interpolate.splrep(x, y, s=len(x))

    bspl1 = scipy.interpolate.BSpline(*tck)
    bspl2 = scipy.interpolate.BSpline(*tck_s)

    xnew = np.arange(0, 9 / 4, 1 / 50) * np.pi
    plt.plot(xnew, np.sin(xnew), '-.', label='sin(x)')
    plt.plot(xnew, bspl1(xnew), '-', label='s=0')
    plt.plot(xnew, bspl2(xnew), '-', label=f's={len(x)}')
    plt.plot(x, y, 'o')
    plt.legend()
    plt.show()


def manipulate_spline_obj():
    x = np.arange(0, 2 * np.pi + np.pi / 4, 2 * np.pi / 8)
    y = np.sin(x) + 0.2 * np.random.uniform(-1, 1, size=(len(x)))
    tck = scipy.interpolate.splrep(x, y, s=4)
    xnew = np.arange(0, 2 * np.pi, np.pi / 50)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)

    plt.figure()
    plt.plot(x, y, 'x')
    plt.plot(xnew, ynew, '--')
    plt.plot(xnew, np.sin(xnew), 'b')
    plt.legend(['Linear', f'Cubic Spline Smooth s={4} ', 'True'])
    plt.axis([-0.05, 6.33, -1.05, 1.05])
    plt.title('Cubic-spline interpolation')
    plt.show()


def kdtreestruct():
    from scipy.spatial import KDTree

    n = 10000  # Number of data points
    data = np.random.rand(6, n)
    data = data.T
    kd_tree = KDTree(data)
    print("Data:\n", data)

    # Query point (6-dimensional)
    query_point = np.random.rand(6)
    distance, index = kd_tree.query(query_point)

    print("\nQuery Point:", query_point)
    print("Nearest Neighbor Index:", index)
    print("Nearest Neighbor:", data[index])
    print("Distance to Nearest Neighbor:", distance)

    k = 3
    distances, indices = kd_tree.query(query_point, k=k)

    print(f"\n{k} Nearest Neighbors Indices:", indices)
    print(f"{k} Nearest Neighbors:\n", data[indices])
    print(f"Distances to {k} Nearest Neighbors:", distances)

if __name__ == "__main__":
    # smoothing_spline_interp()
    kdtreestruct()