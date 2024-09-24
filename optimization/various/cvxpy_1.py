import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

constraints = [x + y == 1, x - y >= 1]

obj = cp.Minimize((x - y) ** 2)

prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)
