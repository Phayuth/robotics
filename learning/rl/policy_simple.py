import random


def deterministic_policy(state):
    if state == "A":
        return "move_right"
    elif state == "B":
        return "move_left"
    else:
        return "stay"


def stochastic_policy(state):
    if state == "A":
        return random.choices(["move_right", "move_left"], weights=[0.8, 0.2])[0]
    elif state == "B":
        return random.choices(["move_right", "move_left"], weights=[0.7, 0.3])[0]
    else:
        return random.choices(["stay", "move_right", "move_left"], weights=[0.8, 0.1, 0.1])[0]


if __name__ == "__main__":
    state = "A"
    print("State:", state)
    print("Simple Deterministic Policy:", deterministic_policy(state))
    print("Simple Stochastic Policy:", stochastic_policy(state))

    state = "B"
    print("\nState:", state)
    print("Simple Deterministic Policy:", deterministic_policy(state))
    print("Simple Stochastic Policy:", stochastic_policy(state))

    state = "C"
    print("\nState:", state)
    print("Simple Deterministic Policy:", deterministic_policy(state))
    print("Simple Stochastic Policy:", stochastic_policy(state))