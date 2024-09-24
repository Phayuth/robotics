from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, Integers, minimize, Binary, ConstraintList

# solving graph shortest path
# https://www.youtube.com/watch?v=7uCx--vUiiI

model = ConcreteModel()

A = ["x12", "x13", "x14", "x23", "x25", "x32", "x34", "x35", "x36", "x43", "x47", "x53", "x56", "x57", "x74", "x75", "x76", "x52"]
w = [35, 30, 20, 8, 12, 8, 9, 10, 20, 9, 15, 10, 5, 20, 15, 20, 5, 12]
model.x = Var(A, within=Binary)

model.obj = Objective(expr=sum(w[i] * model.x[e] for i, e in enumerate(A)), sense=minimize)

model.constraints = ConstraintList()
model.constraints.add(expr=model.x[A[0]] + model.x[A[1]] + model.x[A[2]] == 1)
model.constraints.add(expr=model.x[A[3]] + model.x[A[4]] == model.x[A[0]] + model.x[A[5]] + model.x[A[17]])
model.constraints.add(expr=model.x[A[5]] + model.x[A[6]] + model.x[A[7]] + model.x[A[8]] == model.x[A[1]] + model.x[A[3]] + model.x[A[9]] + model.x[A[11]])
model.constraints.add(expr=model.x[A[9]] + model.x[A[10]] == model.x[A[2]] + model.x[A[6]] + model.x[A[14]])
model.constraints.add(expr=model.x[A[17]] + model.x[A[11]] + model.x[A[12]] + model.x[A[13]] == model.x[A[4]] + model.x[A[7]] + model.x[A[15]])
model.constraints.add(expr=model.x[A[14]] + model.x[A[15]] + model.x[A[16]] == model.x[A[10]] + model.x[A[13]])
model.constraints.add(expr=model.x[A[8]] + model.x[A[12]] + model.x[A[16]] == 1)

opt = SolverFactory("glpk")
result_obj = opt.solve(model, tee=True)

model.pprint()

opt_solution = [model.x[item].value for item in A]
print(f"> opt_solution: {opt_solution}")

xsol = [e for i, e in enumerate(A) if opt_solution[i] == 1]
print(f"> xsol: {xsol}")

objective_obj = model.obj()
print(f"> objective_obj: {objective_obj}")
