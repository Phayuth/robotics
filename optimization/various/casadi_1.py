import casadi as ca
from icecream import ic


x = ca.MX.sym("x")
y = ca.MX.sym("y")
z = ca.MX.sym("z")

nlp = {}
nlp["x"] = ca.vertcat(x, y, z)  # decision vars
nlp["f"] = x**2 + 100 * z**2  # objective
nlp["g"] = z + (1 - x) ** 2 - y  # constraints

# Create solver instance
solver = ca.nlpsol("F", "ipopt", nlp)
res = solver(x0=[2.5, 3.0, 0.75], ubg=0, lbg=0)

# Response
ic(res["x"])
resary = res["x"].__array__()
ic(resary)
