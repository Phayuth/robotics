import numpy as np

# Define the environment: states, actions, transition probabilities, and rewards
states = ['A', 'B', 'C', 'D']
actions = ['left', 'right']
transition_probs = {
    'A': {'left': ('A', 0), 'right': ('B', 1)},  # Moving right from A leads to B with reward 1
    'B': {'left': ('A', 0), 'right': ('C', 0)},  # etc...
    'C': {'left': ('B', 0), 'right': ('D', 10)},
    'D': {'left': ('C', 0), 'right': ('D', 0)}
}
gamma = 0.9  # Discount factor
policy = {s: 'left' for s in states}  # Initialize random policy

# Policy Evaluation function
def policy_evaluation(policy, V, threshold=1e-3):
    while True:
        delta = 0
        for state in states:
            action = policy[state]
            next_state, reward = transition_probs[state][action]
            new_value = reward + gamma * V[next_state]
            delta = max(delta, abs(V[state] - new_value))
            V[state] = new_value
        if delta < threshold:
            break
    return V

# Policy Improvement function
def policy_improvement(V):
    stable_policy = True
    for state in states:
        old_action = policy[state]
        action_values = {}
        for action in actions:
            next_state, reward = transition_probs[state][action]
            action_values[action] = reward + gamma * V[next_state]
        best_action = max(action_values, key=action_values.get)
        policy[state] = best_action
        if old_action != best_action:
            stable_policy = False
    return policy, stable_policy

# Initialize Value function
V = {s: 0 for s in states}

# Policy Iteration loop
while True:
    # Policy Evaluation
    V = policy_evaluation(policy, V)

    # Policy Improvement
    policy, is_stable = policy_improvement(V)

    if is_stable:
        break

# Print the final policy and value function
print("Optimal Policy:", policy)
print("Value Function:", V)
