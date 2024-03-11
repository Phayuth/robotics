import do_mpc
import matplotlib.pyplot as plt
import numpy as np
from casadi import DM, SX


def dynamic_model(current_state, control_input):
    current_state = np.array(current_state)
    control_input = np.array(control_input)
    next_state = current_state + 0.1*control_input
    return next_state


# State model setup
model_type = 'discrete'  # 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

_x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
_u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))

model.set_rhs('x', dynamic_model(_x, _u))
model.setup()

# MPC Controller setup
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 10,
    't_step': 1.0,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)
mpc.settings.supress_ipopt_output()
mterm = _x**2
lterm = _x**2
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(u=1e-2)
mpc.setup()

# Simulator setup
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=1.0)
simulator.setup()

# Run Simulation and Controller
initial_state = np.array([1.0])
mpc.x0 = initial_state
simulator.x0 = initial_state

for _ in range(20):
    u0 = mpc.make_step(simulator.x0)
    print(f"> type(u0): {type(u0)}")
    print(f"> u0: {u0}")
    y_next = simulator.make_step(u0)

# Plot data
mpc_data = mpc.data
time_data = mpc_data['_time']
u_data = mpc_data['_u', 'u']
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
axes[0].plot(time_data, mpc_data['_x', 'x'])
axes[0].set_ylabel('States (x)')
axes[1].plot(time_data, u_data)
axes[1].set_ylabel('Control (u0)')
plt.show()