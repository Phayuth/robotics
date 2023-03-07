import numpy as np
import matplotlib.pyplot as plt
from numpy import cos,sin,sign

def superellipse(theta):
    a = 10 # tune parameter
    b = 10 # tune parameter
    n = 4 # tune parameter

    x = ((abs(cos(theta)))**(2/n))*(a*sign(cos(theta)))
    y = ((abs(sin(theta)))**(2/n))*(b*sign(sin(theta)))

    return x,y

theta = np.linspace(0,2*np.pi,360)
x,y = superellipse(theta)

plt.plot(x,y)
plt.show()