import numpy as np
import matplotlib.pyplot as plt


P0 = np.array([-1, -1]).reshape(2, 1)
P1 = np.array([1, 4]).reshape(2, 1)
P2 = np.array([6, 2]).reshape(2, 1)

t = np.linspace(0, 1, 1000)
C = ((1 - t) ** 2) * P0 + (2 * (1 - t) * t) * P1 + (t**2) * P2
print(f"> C.shape: {C.shape}")

plt.plot(C[0, :], C[1, :])
plt.plot([P0[0, 0], P1[0, 0], P2[0, 0]], [P0[1, 0], P1[1, 0], P2[1, 0]], "b*")
plt.show()
