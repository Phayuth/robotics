from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, Integers, minimize

# Create a concrete model
model = ConcreteModel()

# Define the decision variables
model.x = Var(domain=Integers, initialize=1, bounds=(0, 10))
model.y = Var(domain=Integers, initialize=1, bounds=(0, 20))

# Define the objective function
model.obj = Objective(expr=3*model.x + 2*model.y, sense=minimize)

# Define the constraints
model.constraint1 = Constraint(expr=model.x + model.y == 15)
model.constraint2 = Constraint(expr=model.x >= 3)
model.constraint3 = Constraint(expr=model.y >= 5)

# Select a solver
solver = SolverFactory('glpk')  # You can also use 'cbc' if 'glpk' is not available
model.display()
solver.solve(model)

# Get the results
x_value = model.x()
y_value = model.y()
objective_value = model.obj()

print(f"Optimal value for x: {x_value}")
print(f"Optimal value for y: {y_value}")
print(f"Objective function value: {objective_value}")
