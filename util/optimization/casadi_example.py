from casadi import *
import numpy as np
from icecream import ic
import casadi as ca
import matplotlib.pyplot as plt

# Symbols/expressions
x = MX.sym('x')
y = MX.sym('y')
z = MX.sym('z')
f = x**2 + 100 * z**2
g = z + (1 - x)**2 - y

nlp = {}  # NLP declaration
nlp['x'] = vertcat(x, y, z)  # decision vars
nlp['f'] = f  # objective
nlp['g'] = g  # constraints

# Create solver instance
solver = nlpsol('F', 'ipopt', nlp)

# Solve the problem using a guess
res = solver(x0=[2.5, 3.0, 0.75], ubg=0, lbg=0)

ic(res['x'])

resAsNpArray = res['x'].__array__()
ic(resAsNpArray)





# https://web.casadi.org/docs/#a-simple-test-problem


# x = ca.MX.sym('t')
# f = x**2

# nlp = {}
# nlp['x'] = ca.vertcat(x)
# nlp['f'] = f

# solver = ca.nlpsol('F', 'ipopt', nlp)
# solution = solver(x0=[2.3], ubg=0.0, lbg=10.0)
# ic(solution['x'])








# # Define variables
# x = ca.MX.sym('x')
# y = ca.MX.sym('y')

# # Define an objective function
# objective = x**2 + y**2

# # Create an optimization problem
# opt_problem = {'x': ca.vertcat(x, y), 'f': objective}

# # Create an optimizer
# opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
# solver = ca.nlpsol('solver', 'ipopt', opt_problem, opts)

# # Solve the optimization problem
# initial_guess = [1.0, 2.0]
# solution = solver(x0=initial_guess)

# ic(solution['x'])  # Print the optimal solution