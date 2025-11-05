import numpy as np
import random

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize Q-Learning agent
        
        Q-table stores Q(s,a) values for all state-action pairs
        """
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha      # α: learning rate
        self.gamma = gamma      # γ: discount factor
        self.epsilon = epsilon  # ε: exploration rate
        self.n_actions = n_actions
    
    def choose_action(self, state):
        """
        ε-greedy policy:
        
        π(s) = {  random action           with probability ε
               {  argmax_a Q(s,a)         with probability 1-ε
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update(self, state, action, reward, next_state):
        """
        Q-Learning Update Rule:
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                          └─────────────┬─────────────┘
                                   TD error
        
        Where:
        - Q(s,a) = current Q-value
        - α = learning rate
        - r = immediate reward
        - γ = discount factor
        - max_a' Q(s',a') = max Q-value in next state (bootstrap estimate)
        - TD error = difference between target and current estimate
        """
        current_q = self.q_table[state, action]           # Q(s,a)
        max_next_q = np.max(self.q_table[next_state])    # max_a' Q(s',a')
        
        # TD target: r + γ max_a' Q(s',a')
        # TD error: [r + γ max_a' Q(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state, action] = new_q


class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.goal = size * size - 1  # Bottom-right corner
        self.reset()
    
    def reset(self):
        """Reset to starting position"""
        self.state = 0  # Top-left corner
        return self.state
    
    def _get_position(self, state):
        """Convert state to (row, col)"""
        return state // self.size, state % self.size
    
    def _get_state(self, row, col):
        """Convert (row, col) to state"""
        return row * self.size + col
    
    def step(self, action):
        """
        Actions: 0=up, 1=down, 2=left, 3=right
        Returns: (next_state, reward, done)
        """
        row, col = self._get_position(self.state)
        
        # Take action
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)
        
        # Update state
        self.state = self._get_state(row, col)
        
        # Calculate reward
        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1  # Small penalty for each step
            done = False
        
        return self.state, reward, done
    
    def render(self, q_table=None):
        """Visualize the grid"""
        for row in range(self.size):
            for col in range(self.size):
                state = self._get_state(row, col)
                if state == self.state:
                    print("A", end=" ")  # Agent
                elif state == self.goal:
                    print("G", end=" ")  # Goal
                else:
                    print(".", end=" ")
            print()
        print()


def train():
    # Create environment and agent
    env = GridWorld(size=4)
    agent = QLearning(n_states=env.n_states, n_actions=env.n_actions, 
                      alpha=0.1, gamma=0.9, epsilon=0.2)
    
    # Training
    episodes = 500
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        rewards_per_episode.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}")
    
    return env, agent


def test_agent(env, agent):
    """Test the trained agent"""
    print("\n=== Testing Trained Agent ===")
    agent.epsilon = 0  # No exploration, pure exploitation
    
    state = env.reset()
    env.render()
    
    steps = 0
    max_steps = 20
    
    while steps < max_steps:
        action = agent.choose_action(state)
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"Step {steps + 1}: Action = {action_names[action]}")
        
        next_state, reward, done = env.step(action)
        env.render()
        
        state = next_state
        steps += 1
        
        if done:
            print(f"Goal reached in {steps} steps!")
            break
    
    print("\nLearned Q-Table (sample):")
    print(agent.q_table)  # Show first row


if __name__ == "__main__":
    env, agent = train()
    test_agent(env, agent)