import numpy as np
import scipy.optimize as sop
import matplotlib.pyplot as plt

# linear regression slope variable

def model(x):
    areal = 3.2
    return areal * x


noise = 1.2 * np.random.uniform(-1, 1, 100)
x = np.linspace(0, 10, 100)
y = model(x)

ynoise = y + noise


def objective_function(aopt):
    yhat = aopt * x
    err = yhat - ynoise
    cost = np.sum(np.square(np.abs(err)))
    return cost


initial_guess = [6.1]
result = sop.minimize(objective_function, initial_guess, options={"disp": True})
print(f"> result: {result}")

aopt = result.x
yest = aopt * x

plt.plot(x, y, "r", label="model")
plt.plot(x, ynoise, "b.", label="model noise")
plt.plot(x, yest, "g", label="model estimated")
plt.legend()
plt.show()

a_range = np.linspace(-3.0, 10.0, 100)
objplot = [objective_function(ai) for ai in a_range]

plt.plot(a_range, objplot)
plt.plot(initial_guess, objective_function(initial_guess[0]), "gx")
plt.plot(aopt, objective_function(aopt), "r*")
plt.show()
