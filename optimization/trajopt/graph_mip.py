from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, Integers, minimize, Binary, ConstraintList

# solving graph shortest path
# https://www.youtube.com/watch?v=7uCx--vUiiI

model = ConcreteModel()

# notation
# x(start, end). ex x12 is edge start from 1 and end at 2.
# xij = 1 if edge ij is in the shortest path.
# xij = 0 otherwise.
# w = weight of edge.
# x12 weight is 35 means it cost 35 to go from 1 to 2.
A = ["x12", "x13", "x14", "x23", "x25", "x32", "x34", "x35", "x36", "x43", "x47", "x53", "x56", "x57", "x74", "x75", "x76", "x52"]
w = [35, 30, 20, 8, 12, 8, 9, 10, 20, 9, 15, 10, 5, 20, 15, 20, 5, 12]
model.x = Var(A, within=Binary)

model.obj = Objective(expr=sum(w[i] * model.x[e] for i, e in enumerate(A)), sense=minimize)

model.constraints = ConstraintList()

# start and end constraints
model.constraints.add(expr=model.x[A[0]] + model.x[A[1]] + model.x[A[2]] == 1) # we must start from node 1 so sum of x1i = 1
model.constraints.add(expr=model.x[A[8]] + model.x[A[12]] + model.x[A[16]] == 1) # we must end at node 6 so sum of xi6 = 1

# define the constraints for bidirectional edges
# sum of edge start = sum of edge end
# ex: x23 + x25 = x12 + x32 + x52. sum of edge start at node 2 = sum of edge end at node 2
model.constraints.add(expr=model.x[A[3]] + model.x[A[4]] == model.x[A[0]] + model.x[A[5]] + model.x[A[17]])
model.constraints.add(expr=model.x[A[5]] + model.x[A[6]] + model.x[A[7]] + model.x[A[8]] == model.x[A[1]] + model.x[A[3]] + model.x[A[9]] + model.x[A[11]])
model.constraints.add(expr=model.x[A[9]] + model.x[A[10]] == model.x[A[2]] + model.x[A[6]] + model.x[A[14]])
model.constraints.add(expr=model.x[A[17]] + model.x[A[11]] + model.x[A[12]] + model.x[A[13]] == model.x[A[4]] + model.x[A[7]] + model.x[A[15]])
model.constraints.add(expr=model.x[A[14]] + model.x[A[15]] + model.x[A[16]] == model.x[A[10]] + model.x[A[13]])

opt = SolverFactory("glpk")
result_obj = opt.solve(model, tee=True)

model.pprint()

opt_solution = [model.x[item].value for item in A]
print(f"> opt_solution: {opt_solution}")

xsol = [e for i, e in enumerate(A) if opt_solution[i] == 1]
print(f"> xsol: {xsol}")

objective_obj = model.obj()
print(f"> objective_obj: {objective_obj}")
