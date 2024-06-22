import casadi as ca
from icecream import ic

x = ca.MX.sym("t")
f = x**2

nlp = {}
nlp["x"] = ca.vertcat(x)
nlp["f"] = f
solver = ca.nlpsol("F", "ipopt", nlp)
solution = solver(x0=[2.3], ubg=0.0, lbg=10.0)

ic(solution["x"])
