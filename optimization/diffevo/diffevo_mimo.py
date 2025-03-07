import numpy as np
from scipy.optimize import differential_evolution

def objective_function(params):
    theta1, theta2, alpha, r, v_max = params
    x = np.linspace(0, 10, 100)
    y_true1 = np.sin(theta1*x) + np.cos(theta2*x) + alpha*np.exp(-r*x)
    y_true2 = v_max/(1 + np.exp(-r*(x - alpha)))
    y_pred1 = np.sin(params[0]*x) + np.cos(params[1]*x) + params[2]*np.exp(-params[3]*x)
    y_pred2 = params[4]/(1 + np.exp(-params[3]*(x - params[2])))
    error1 = np.sum((y_true1 - y_pred1)**2)
    error2 = np.sum((y_true2 - y_pred2)**2)
    error = error1 + error2
    return error

bounds = [(0, np.pi), (0, np.pi), (0, 10), (0, 1), (0, 10)]

result = differential_evolution(objective_function, bounds)

print('Optimized parameters:', result.x)
print('Objective function value:', result.fun)