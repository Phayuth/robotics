import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

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


K = 7
N = 10
d = 2


# compute joint displacement ------------------------------
noisy = np.random.randint(low=0, high=10, size=(K, N, d))
ic(noisy.shape)
ic(noisy)

d = np.diff(noisy, axis=1)
ic(d)
ic(d.shape)

e = np.diff(noisy, axis=1, prepend=0)
e[:, 0, :] = 0.0
ic(e)
ic(e.shape)

f = np.abs(e).sum(axis=2).T
# f = e.sum(axis=2).T
ic(f)
ic(f.shape)


# # compute P-----------------------------------------------
# S = np.random.randint(low=0,high=10,size=(N, K))
# ic(S)

# # K trajectory wise
# Smin = S.min(axis=1, keepdims=True)
# Smax = S.max(axis=1, keepdims=True)

# ic(Smin)
# ic(Smax)
# eS = Smax - Smin
# ic(eS)

# ee = (S - Smin) / eS
# ic(ee)

# exp = np.exp(-10*ee)
# ic(exp)


# nn = 15
# epsilon = np.random.randint(low=0, high=10, size=nn)
# ic(epsilon)

# S = np.random.uniform(0, 10, size=(nn))
# ic(S)

# beta = 1.0
# softmax = np.exp(beta * S) / np.sum(np.exp(beta * S))
# ic(softmax)

# h = 1.0
# Smin = S.min()  # K trajectory wise
# Smax = S.max()
# softmax_inv = np.exp(-h * (S - Smin) / (Smax - Smin))
# ic(softmax_inv)


# def stable_softmax(x):
#     z = x - max(x)
#     numerator = np.exp(z)
#     denominator = np.sum(numerator)
#     softmax = numerator/denominator
#     return softmax


# softmaxstable = stable_softmax(S)
# ic(softmaxstable)
