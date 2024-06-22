import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
# https://towardsdatascience.com/kernel-density-estimation-explained-step-by-step-7cc5b5bc4517


def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N) :] += 5
    return x


x = make_data(1000)
hist = plt.hist(x, bins=30)
plt.show()


# density, bins, patches = hist
# widths = bins[1:] - bins[:-1]
# (density * widths).sum()


x = make_data(20)
x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), "|k", markeredgewidth=1)

plt.axis([-4, 8, -0.2, 8])
plt.show()


x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), "|k", markeredgewidth=1)

plt.axis([-4, 8, -0.2, 5])
plt.show()
