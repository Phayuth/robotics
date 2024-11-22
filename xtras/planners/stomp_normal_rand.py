import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=2000, suppress=True)
np.random.seed(9)


def single_normal_rand():
    mu = 0
    sigma = 1
    xrand = np.random.normal(mu, sigma)


def multivariate_normal_rand():
    mus = np.array([2, 0])
    sigx = 1
    sigy = 1
    sigmas = np.array([[sigx**2, 0.5], [0.5, sigy**2]])
    x, y = np.random.multivariate_normal(mus, sigmas, size=(100000)).T
    print(f"> x.shape: {x.shape}")
    print(f"> y.shape: {y.shape}")

    plt.plot(x, y, "b*")
    plt.axis("equal")
    plt.show()
