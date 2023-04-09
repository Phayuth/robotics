import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the parabolic equation
def parabolic(x, a, b, c):
    return a*x**2 + b*x + c

# Generate some sample data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# y = np.array([1.2, 2.4, 4.2, 6.6, 9.6, 13.2])
y = np.array([1.2, 2.4, 3.6, 4.8, 9.6, 13.2, 9.6, 6.6])


# Fit the parabolic equation to the data
popt, pcov = curve_fit(parabolic, x, y)

# Print the values of a, b, and c
print("a = ", popt[0])
print("b = ", popt[1])
print("c = ", popt[2])

# Plot the original data and the fitted curve
plt.plot(x, y, 'bo', label='Original Data')
plt.plot(x, parabolic(x, *popt), 'r-', label='Fitted Curve')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
