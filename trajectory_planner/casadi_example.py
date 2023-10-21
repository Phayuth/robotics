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
F = nlpsol('F', 'ipopt', nlp)

# Solve the problem using a guess
F(x0=[2.5, 3.0, 0.75], ubg=0, lbg=0)







# https://web.casadi.org/docs/#a-simple-test-problem 


# x = ca.MX.sym('t')
# f = x**2 #x**2 - 2*x + 1

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






# Number of variables
n = 5

# Define variables as a vector
x = ca.MX.sym('x', n)
y = ca.MX.sym('y', n)

# Define the summation of square root terms in the objective function
squared_norms = (x**2 + y**2)
objective = ca.sum1(ca.sqrt(squared_norms))

# Create constraints
inequality_constraint = ca.sum1(x + y) - 1
equality_constraint = ca.sum1(x - y) - 1

# Define variable bounds for each variable
variable_bounds = {
    'lbx': [0] * (2 * n),  # Lower bounds for x and y
    'ubx': [2] * (2 * n)   # Upper bounds for x and y
}

# Create an optimization problem
opt_problem = {
    'x': ca.vertcat(x, y),
    'f': objective,
    'g': ca.vertcat(inequality_constraint, equality_constraint)
}

# Create an optimizer
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
solver = ca.nlpsol('solver', 'ipopt', opt_problem, opts)

# Set the initial guess
initial_guess = [1.0] * (2 * n)
print(f"==>> initial_guess: {initial_guess}")

# Solve the optimization problem
solution = solver(x0=initial_guess, **variable_bounds)

# Extract the results
optimal_x = solution['x']
# optimal_objective = solution['f']
# constrained_inequality = solution['g'][0]
# constrained_equality = solution['g'][1]

# print("Optimal Solution:")
xopt = optimal_x[:n].toarray()
yopt = optimal_x[n:].toarray()
print("x =", xopt)
print("y =", yopt)
# print("Optimal Objective Value:", optimal_objective)
# print("Inequality Constraint Value:", constrained_inequality)
# print("Equality Constraint Value:", constrained_equality)


# Plot x and y pairs
plt.scatter(xopt, yopt, label="Optimal Solution", color="red", marker="o")
plt.xlabel('x')
plt.ylabel('y')

# Optionally, you can add labels or other elements to the plot if needed.
# For example, you can add a grid or legend.

plt.grid()
plt.legend()

# Show the plot
plt.show()


