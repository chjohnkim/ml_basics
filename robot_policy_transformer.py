"""
Robot Policy Transformer for Imitation Learning

This example demonstrates a transformer-based policy network for robot control
using behavior cloning (imitation learning). The policy learns to predict
actions from state observations by imitating expert demonstrations.

Task: Simple 2D robot arm reaching task
- State: joint angles + end-effector position + goal position (8D)
- Action: joint velocity commands (2D)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        pe = torch.zeros(1, max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[:, pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
                else:
                    pe[:, pos, i] = np.cos(pos / (10000 ** (2 * (i - 1) / d_model)))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size, causal=True):
        super().__init__()
        self.head_size = head_size
        self.causal = causal
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention = torch.matmul(q, k.transpose(-2, -1)) / self.head_size ** 0.5

        # Causal masking for autoregressive policy
        if self.causal:
            seq_len = x.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attention = attention.masked_fill(mask, float('-inf'))

        attention = torch.softmax(attention, dim=-1)
        attention = torch.matmul(attention, v)
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        self.head_size = d_model // n_heads
        self.W_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.head_size, causal) for _ in range(n_heads)
        ])

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4, causal=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model),
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RobotPolicyTransformer(nn.Module):
    """
    Transformer-based policy for imitation learning.

    Takes a sequence of (state, action) pairs and predicts the next action.
    Uses causal attention so each position can only attend to previous positions.
    """
    def __init__(self, state_dim, action_dim, d_model, n_heads, n_layers, max_seq_length):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Embeddings for states and actions
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)

        # Learned token type embeddings (state vs action)
        self.token_type_embed = nn.Embedding(2, d_model)  # 0 for state, 1 for action

        self.pe = PositionalEncoding(d_model, max_seq_length * 2)  # *2 for interleaved state-action
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, r_mlp=4, causal=True) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Action prediction head
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, states, actions=None):
        """
        Args:
            states: (B, T, state_dim) - sequence of states
            actions: (B, T-1, action_dim) - sequence of actions (optional during inference)
        Returns:
            predicted_actions: (B, T, action_dim) - predicted action for each state
        """
        B, T, _ = states.shape
        device = states.device

        # Embed states
        state_tokens = self.state_embed(states)  # (B, T, d_model)
        state_tokens = state_tokens + self.token_type_embed(torch.zeros(T, dtype=torch.long, device=device))

        if actions is not None:
            # Training mode: interleave states and actions
            # Sequence: s0, a0, s1, a1, s2, a2, ...
            action_tokens = self.action_embed(actions)  # (B, T-1, d_model)
            action_tokens = action_tokens + self.token_type_embed(torch.ones(T-1, dtype=torch.long, device=device))

            # Interleave: [s0, a0, s1, a1, ...]
            seq_len = T + T - 1
            tokens = torch.zeros(B, seq_len, self.d_model, device=device)
            tokens[:, 0::2, :] = state_tokens  # states at even positions
            tokens[:, 1::2, :] = action_tokens  # actions at odd positions
        else:
            # Inference mode: just states
            tokens = state_tokens

        # Add positional encoding
        tokens = self.pe(tokens)

        # Transformer
        tokens = self.transformer(tokens)
        tokens = self.ln_f(tokens)

        if actions is not None:
            # Extract predictions at state positions (predict action after seeing state)
            state_outputs = tokens[:, 0::2, :]  # (B, T, d_model)
        else:
            state_outputs = tokens

        # Predict actions
        predicted_actions = self.action_head(state_outputs)
        return predicted_actions


class ExpertDemonstrationDataset(Dataset):
    """
    Synthetic dataset simulating expert demonstrations for a 2D reaching task.

    The robot has 2 joints and needs to reach a goal position.
    Expert policy: simple proportional control towards the goal.
    """
    def __init__(self, n_episodes=1000, episode_length=20, noise_std=0.01):
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.noise_std = noise_std

        # Generate all demonstrations
        self.states, self.actions = self._generate_demonstrations()

    def _forward_kinematics(self, joint_angles):
        """Compute end-effector position from joint angles (2-link arm)."""
        l1, l2 = 1.0, 1.0  # link lengths
        x = l1 * np.cos(joint_angles[0]) + l2 * np.cos(joint_angles[0] + joint_angles[1])
        y = l1 * np.sin(joint_angles[0]) + l2 * np.sin(joint_angles[0] + joint_angles[1])
        return np.array([x, y])

    def _expert_policy(self, joint_angles, ee_pos, goal_pos):
        """Simple proportional control expert policy."""
        # Direction to goal in task space
        direction = goal_pos - ee_pos

        # Simple Jacobian transpose control
        j1, j2 = joint_angles
        l1, l2 = 1.0, 1.0

        # Jacobian
        J = np.array([
            [-l1*np.sin(j1) - l2*np.sin(j1+j2), -l2*np.sin(j1+j2)],
            [l1*np.cos(j1) + l2*np.cos(j1+j2), l2*np.cos(j1+j2)]
        ])

        # Joint velocities via Jacobian transpose
        action = J.T @ direction * 0.5
        action = np.clip(action, -0.5, 0.5)

        # Add small noise to make learning more robust
        action += np.random.randn(2) * self.noise_std
        return action

    def _generate_demonstrations(self):
        """Generate expert demonstrations."""
        all_states = []
        all_actions = []

        for _ in range(self.n_episodes):
            # Random initial joint angles
            joint_angles = np.random.uniform(-np.pi/2, np.pi/2, 2)

            # Random goal position (reachable workspace)
            goal_angle = np.random.uniform(0, 2*np.pi)
            goal_dist = np.random.uniform(0.5, 1.8)
            goal_pos = np.array([goal_dist * np.cos(goal_angle), goal_dist * np.sin(goal_angle)])

            episode_states = []
            episode_actions = []

            for t in range(self.episode_length):
                ee_pos = self._forward_kinematics(joint_angles)

                # State: [joint_angles, ee_pos, goal_pos]
                state = np.concatenate([joint_angles, ee_pos, goal_pos])
                episode_states.append(state)

                # Expert action
                action = self._expert_policy(joint_angles, ee_pos, goal_pos)
                episode_actions.append(action)

                # Update joint angles
                joint_angles = joint_angles + action * 0.1  # dt = 0.1
                joint_angles = np.clip(joint_angles, -np.pi, np.pi)

            all_states.append(np.array(episode_states))
            all_actions.append(np.array(episode_actions))

        return np.array(all_states), np.array(all_actions)

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        states = torch.FloatTensor(self.states[idx])
        actions = torch.FloatTensor(self.actions[idx])
        return states, actions


def train(model, train_loader, test_loader, optimizer, epochs, device):
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        training_loss = 0

        for states, actions in tqdm(train_loader, desc=f"Epoch {epoch}"):
            states, actions = states.to(device), actions.to(device)

            optimizer.zero_grad()

            # Predict actions given states and previous actions
            # Input: all states, actions[:-1] (all but last action)
            # Target: all actions
            predicted_actions = model(states, actions[:, :-1, :])

            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        avg_train_loss = training_loss / len(train_loader)
        print(f"Epoch {epoch}: Training Loss: {avg_train_loss:.6f}")

        # Evaluation
        model.eval()
        testing_loss = 0

        with torch.no_grad():
            for states, actions in test_loader:
                states, actions = states.to(device), actions.to(device)
                predicted_actions = model(states, actions[:, :-1, :])
                loss = criterion(predicted_actions, actions)
                testing_loss += loss.item()

        avg_test_loss = testing_loss / len(test_loader)
        print(f"Epoch {epoch}: Testing Loss: {avg_test_loss:.6f}")

    return model


def evaluate_policy(model, device, n_episodes=10):
    """Evaluate learned policy by rolling out trajectories."""
    model.eval()

    dataset = ExpertDemonstrationDataset(n_episodes=1, episode_length=1)

    total_distance = 0

    for ep in range(n_episodes):
        # Random initial state
        joint_angles = np.random.uniform(-np.pi/2, np.pi/2, 2)
        goal_angle = np.random.uniform(0, 2*np.pi)
        goal_dist = np.random.uniform(0.5, 1.8)
        goal_pos = np.array([goal_dist * np.cos(goal_angle), goal_dist * np.sin(goal_angle)])

        states_history = []

        for t in range(20):
            ee_pos = dataset._forward_kinematics(joint_angles)
            state = np.concatenate([joint_angles, ee_pos, goal_pos])
            states_history.append(state)

            # Get action from policy
            states_tensor = torch.FloatTensor(states_history).unsqueeze(0).to(device)

            with torch.no_grad():
                predicted_actions = model(states_tensor)
                action = predicted_actions[0, -1, :].cpu().numpy()

            # Update state
            joint_angles = joint_angles + action * 0.1
            joint_angles = np.clip(joint_angles, -np.pi, np.pi)

        # Final distance to goal
        final_ee_pos = dataset._forward_kinematics(joint_angles)
        distance = np.linalg.norm(final_ee_pos - goal_pos)
        total_distance += distance

    avg_distance = total_distance / n_episodes
    print(f"\nEvaluation: Average final distance to goal: {avg_distance:.4f}")
    return avg_distance


if __name__ == "__main__":
    # Hyperparameters
    state_dim = 6  # 2 joint angles + 2 ee pos + 2 goal pos
    action_dim = 2  # 2 joint velocities
    d_model = 64
    n_heads = 4
    n_layers = 3
    max_seq_length = 20
    batch_size = 32
    epochs = 20
    lr = 1e-3

    # Create datasets
    print("Generating expert demonstrations...")
    train_dataset = ExpertDemonstrationDataset(n_episodes=5000, episode_length=max_seq_length)
    test_dataset = ExpertDemonstrationDataset(n_episodes=500, episode_length=max_seq_length)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = RobotPolicyTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_length=max_seq_length
    )
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Train
    print("\nTraining...")
    model = train(model, train_loader, test_loader, optimizer, epochs, device)

    # Evaluate
    print("\nEvaluating learned policy...")
    evaluate_policy(model, device, n_episodes=20)
