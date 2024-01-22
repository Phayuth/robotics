import numpy as np
import inspect


class EulerNumericalIntegrator:
    """
    [Summary] : Numerical Integrator Based On Euler Method

    [Method] :

        - `dynamicFunction(previousState, systemInput)` : callable function for system dynamic update. It must accept 2 argument.
        - `controlFunction(previousState, desiredState)` : callable function for controller calculation for systemInput. It must accept 2 argument.
        - `desiredFunction(previousState, time)` : callable function for desiredState calculation. It must accept 2 argument.
        - initialState : system state at the initial time. It must given in shape (n,1) where n is system output dimension.
        - tSpan : simulation time span given as tuple of (0, endTime).
        - dt : time step, typically given as 0.01s.

        StateIndex

        - state [0] + time [1]    -> desired [1]
        - state [0] + desired [1] -> control [1]
        - state [0] + control [1] -> state [1]
    """

    def __init__(self, dynamicFunction, controlFunction, desiredFunction, initialState, tSpan, dt) -> None:
        # timestep handling
        self.dt = dt
        self.numSteps = int((tSpan[1] - tSpan[0]) / self.dt) + 1
        self.timeSteps = np.linspace(tSpan[0], tSpan[1], self.numSteps)

        # callable function handling
        self.dynamicFunction = dynamicFunction
        self.controlFunction = controlFunction
        self.desiredFunction = desiredFunction

        # verify system
        self.system_check(initialState, self.timeSteps[0])

        # states control desired handling
        self.states = np.zeros((self.dynamicDimension, self.numSteps))
        self.states[:, 0, np.newaxis] = initialState
        self.controls = np.zeros((self.controlDimension, self.numSteps))
        self.desireds = np.zeros((self.desiredDimension, self.numSteps))

    def system_check(self, state, time):
        # check system dimension
        tempDesiredState = self.desiredFunction(state, time)
        tempControl = self.controlFunction(state, tempDesiredState)

        self.dynamicDimension = len(state)
        self.desiredDimension = len(tempDesiredState)
        self.controlDimension = len(tempControl)

        # check function argument
        assert len(inspect.signature(self.dynamicFunction).parameters) == 2, "Dynamic function must have 2 argument, (State, time)"
        assert len(inspect.signature(self.controlFunction).parameters) == 2, "Control function must have 2 argument, (State, input)"
        assert len(inspect.signature(self.desiredFunction).parameters) == 2, "Desired function must have 2 argument, (State, desiredState)"

    def simulation(self):
        for i in range(1, self.numSteps):
            self.desireds[:, i, np.newaxis] = self.desiredFunction(self.states[:, i - 1, np.newaxis], self.timeSteps[i])
            self.controls[:, i, np.newaxis] = self.controlFunction(self.states[:, i - 1, np.newaxis], self.desireds[:, i, np.newaxis])
            self.states[:, i, np.newaxis] = self.states[:, i - 1, np.newaxis] + self.dynamicFunction(self.states[:, i - 1, np.newaxis], self.controls[:, i, np.newaxis]) * self.dt

        return self.timeSteps, self.states, self.desireds, self.controls


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simulation parameters
    initialState = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    tSpan = (0, 10)  # Simulation time span
    dt = 0.01  # Time step

    def desired_setpoint(previousState, time):
        return np.array([10.0, 10.0, 0.0]).reshape(3, 1)

    def dynamic_system(previousState, input):
        return np.array([[input[0, 0] * np.cos(previousState[2, 0])],
                         [input[0, 0] * np.sin(previousState[2, 0])],
                         [input[1, 0]]])

    def control(currentState, desiredState):
        u = np.array([[0.6], [0.4]])
        return u

    intg = EulerNumericalIntegrator(dynamic_system, control, desired_setpoint, initialState, tSpan, dt)
    timeSteps, states, desired, controls = intg.simulation()
    print(f"> timeSteps.shape: {timeSteps.shape}")
    print(f"> states.shape: {states.shape}")
    print(f"> desired.shape: {desired.shape}")
    print(f"> controls.shape: {controls.shape}")

    # Plot the results
    plt.plot(states[0, :], states[1, :])
    plt.plot(timeSteps, states[0, :], label='x')
    plt.plot(timeSteps, states[1, :], label='y')
    plt.plot(timeSteps, states[2, :], label=r'$\theta$')
    plt.legend()
    plt.show()