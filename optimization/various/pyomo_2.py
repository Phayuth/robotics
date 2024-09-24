from pyomo.environ import *

# knapsack problem

A = ["hammer", "wrench", "screwdriver", "towel"]
earn = {"hammer": 8, "wrench": 3, "screwdriver": 6, "towel": 11}
weight = {"hammer": 5, "wrench": 7, "screwdriver": 4, "towel": 3}
W_max = 14
model = ConcreteModel()
model.x = Var(A, within=Binary)
model.value = Objective(expr=sum(earn[i] * model.x[i] for i in A), sense=maximize)
model.weight = Constraint(expr=sum(weight[i] * model.x[i] for i in A) <= W_max)

opt = SolverFactory("glpk")
result_obj = opt.solve(model, tee=True)

model.pprint()

optimal_values = {item: model.x[item].value for item in A}
print(f"> optimal_values: {optimal_values}")

print("Optimal item selection:")
for item, value in optimal_values.items():
    print(f"{item}: {'Selected' if value == 1 else 'Not selected'}")

objective_value = model.value()
print(f"Total earnings: {objective_value}")
