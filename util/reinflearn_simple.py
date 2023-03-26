import random

# Define the environment
class Environment:
    def __init__(self):
        self.state = 0
        self.actions = [0, 1]
        self.rewards = [0, 1]

    def step(self, action):
        if action in self.actions:
            if self.state == 0 and action == 0:
                reward = self.rewards[1]
                self.state = 1
            elif self.state == 0 and action == 1:
                reward = self.rewards[0]
                self.state = 0
            else:
                reward = self.rewards[0]
        else:
            reward = self.rewards[0]

        return self.state, reward

# Define the agent
class Agent:
    def __init__(self, learning_rate, discount_factor, exploration_rate):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {(0, 0): 0, (0, 1): 0}

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice([0, 1])
        else:
            q_values = [self.q_table[(state, a)] for a in [0, 1]]
            action = [i for i, q in enumerate(q_values) if q == max(q_values)]
            action = random.choice(action)

        return action

    def learn(self, state, action, reward, next_state):
        next_q_value = max([self.q_table[(next_state, a)] for a in [0, 1]])
        current_q_value = self.q_table[(state, action)]
        td_error = reward + self.discount_factor * next_q_value - current_q_value
        self.q_table[(state, action)] += self.learning_rate * td_error

# Run the simulation
env = Environment()
agent = Agent(learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1)

for i in range(100):
    state = env.state
    action = agent.choose_action(state)
    next_state, reward = env.step(action)
    agent.learn(state, action, reward, next_state)

    print('Episode {}: State = {}, Action = {}, Reward = {}'.format(i+1, state, action, reward))

print('Final Q-Table: {}'.format(agent.q_table))