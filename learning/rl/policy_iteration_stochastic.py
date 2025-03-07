import numpy as np
import random

# Define the environment: states, actions, transition probabilities, and rewards
states = ['A', 'B', 'C', 'D']
actions = ['left', 'right']
transition_probs = {
    'A': {'left': ('A', 0), 'right': ('B', 1)},
    'B': {'left': ('A', 0), 'right': ('C', 0)},
    'C': {'left': ('B', 0), 'right': ('D', 10)},
    'D': {'left': ('C', 0), 'right': ('D', 0)}
}
gamma = 0.9  # Discount factor
policy = {s: {a: 0.5 for a in actions} for s in states}  # Initialize equal probability policy

# Policy Evaluation function for stochastic policy
def policy_evaluation(policy, V, threshold=1e-3):
    while True:
        delta = 0
        for state in states:
            new_value = 0
            for action, action_prob in policy[state].items():
                next_state, reward = transition_probs[state][action]
                new_value += action_prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[state] - new_value))
            V[state] = new_value
        if delta < threshold:
            break
    return V

# Policy Improvement function using softmax
def policy_improvement(V, tau=1.0):
    stable_policy = True
    for state in states:
        action_values = {}
        for action in actions:
            next_state, reward = transition_probs[state][action]
            action_values[action] = reward + gamma * V[next_state]

        # Calculate new action probabilities using softmax
        action_probs = {a: np.exp(action_values[a] / tau) for a in actions}
        total_prob = sum(action_probs.values())
        new_policy = {a: action_probs[a] / total_prob for a in actions}

        # Check if the policy is stable
        if policy[state] != new_policy:
            stable_policy = False
        policy[state] = new_policy

    return policy, stable_policy

# Initialize Value function
V = {s: 0 for s in states}

# Stochastic Policy Iteration loop
while True:
    # Policy Evaluation
    V = policy_evaluation(policy, V)

    # Policy Improvement
    policy, is_stable = policy_improvement(V)

    if is_stable:
        break

# Print the final stochastic policy and value function
print("Optimal Stochastic Policy:")
for state, actions in policy.items():
    print(f"{state}: {actions}")
print("Value Function:", V)
