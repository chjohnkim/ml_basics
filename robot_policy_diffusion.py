"""
Robot Policy Diffusion for Imitation Learning

This example demonstrates a diffusion-based policy network for robot control
using behavior cloning (imitation learning). The policy learns to generate
actions from state observations using denoising diffusion.

Diffusion models iteratively denoise random noise into actions, conditioned
on the current state. This allows modeling multimodal action distributions.

Task: Simple 2D robot arm reaching task
- State: joint angles + end-effector position + goal position (6D)
- Action: joint velocity commands (2D)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        # Positional encoding from "Attention is All You Need"
        # PE(t, 2i) = sin(t / 10000^(2i/d))
        # PE(t, 2i+1) = cos(t / 10000^(2i/d))
        # where t is the timestep and i is the dimension index
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DenoisingNetwork(nn.Module):
    """
    MLP-based denoising network for the diffusion model.

    Takes noisy actions and timestep, conditioned on state observations,
    and predicts the noise to remove.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, time_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Combine noisy action + state observation + time embedding
        input_dim = action_dim + state_dim + time_dim

        # Denoising network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, noisy_action, state, timestep):
        """
        Predict noise ε_θ(x_t, s, t) given:
            - noisy action x_t
            - state observation s (conditioning)
            - timestep t

        The network learns to estimate what noise was added to create x_t,
        which allows us to reverse the diffusion process during sampling.

        Args:
            noisy_action: (B, action_dim) - noisy action x_t
            state: (B, state_dim) - state observation s
            timestep: (B,) - diffusion timestep t
        Returns:
            noise_pred: (B, action_dim) - predicted noise ε_θ(x_t, s, t)
        """
        # Get time embedding: converts timestep into continuous representation
        time_emb = self.time_mlp(timestep)  # (B, time_dim)

        # Concatenate all inputs: [x_t, s, emb(t)]
        x = torch.cat([noisy_action, state, time_emb], dim=-1)

        # Predict noise: ε_θ(x_t, s, t)
        noise_pred = self.net(x)
        return noise_pred


class RobotPolicyDiffusion(nn.Module):
    """
    Diffusion-based policy for imitation learning using DDPM.

    KEY EQUATIONS:

    Forward diffusion (training):
        q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε, where ε ~ N(0, I)

    Reverse diffusion (sampling):
        p_θ(x_{t-1} | x_t, s) = N(x_{t-1}; μ_θ(x_t, s, t), σ_t²·I)
        μ_θ = (1/√α_t)·(x_t - (β_t/√(1-ᾱ_t))·ε_θ(x_t, s, t))

    Training objective:
        L = E_{t,x_0,ε}[||ε - ε_θ(x_t, s, t)||²]

    where:
        x_0: clean action, x_t: noisy action at step t
        s: state observation (conditioning)
        ε_θ: neural network that predicts noise
        α_t = 1 - β_t, ᾱ_t = ∏_{i=1}^t α_i
        β_t: noise schedule

    During training, we add noise to actions and learn to predict that noise.
    During inference, we start from random noise and iteratively denoise to
    produce an action conditioned on the current state.
    """
    def __init__(self, state_dim, action_dim, n_diffusion_steps=100, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_diffusion_steps = n_diffusion_steps

        # Denoising network
        self.denoiser = DenoisingNetwork(state_dim, action_dim, hidden_dim)

        # Diffusion schedule (linear schedule)
        # β_t: variance schedule (amount of noise added at each step)
        self.register_buffer('betas', self._linear_beta_schedule(n_diffusion_steps))
        # α_t = 1 - β_t: retention rate at each step
        self.register_buffer('alphas', 1.0 - self.betas)
        # ᾱ_t = ∏_{i=1}^{t} α_i: cumulative product of alphas
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        # √ᾱ_t: coefficient for the original signal in forward diffusion
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        # √(1 - ᾱ_t): coefficient for the noise in forward diffusion
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
    def _linear_beta_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        """
        Linear noise schedule: β_t increases linearly from β_start to β_end.

        β_t determines how much noise is added at each step:
        - Small β_t (e.g., 1e-4): small noise, slow diffusion
        - Large β_t (e.g., 0.02): more noise, faster diffusion

        The schedule controls the trade-off between sample quality and
        number of denoising steps needed.
        """
        return torch.linspace(beta_start, beta_end, timesteps)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to actions.

        Equation: q(x_t | x_0) = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
        where:
            x_t: noisy action at timestep t
            x_0: original clean action
            ᾱ_t: cumulative product of alphas up to step t
            ε ~ N(0, I): Gaussian noise

        This closed-form formula allows us to sample any noisy version of x_0
        directly without iterating through all previous timesteps.

        Args:
            x_start: (B, action_dim) - clean actions
            t: (B,) - timesteps
            noise: (B, action_dim) - noise to add (optional)
        Returns:
            noisy_action: (B, action_dim)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]

        # x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, actions, states):
        """
        Training forward pass - implements the DDPM training algorithm.

        Training procedure:
        1. Sample random timestep t ~ Uniform(0, T) for each example
        2. Sample noise ε ~ N(0, I)
        3. Create noisy action: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        4. Predict noise: ε_θ(x_t, s, t) using the denoising network
        5. Compute loss: ||ε - ε_θ(x_t, s, t)||²

        By learning to predict the noise at random timesteps, the network
        learns to denoise, which enables the reverse sampling process.

        Args:
            actions: (B, action_dim) - ground truth actions
            states: (B, state_dim) - state observations
        Returns:
            loss: scalar - diffusion loss
        """
        B = actions.shape[0]
        device = actions.device

        # Sample random timesteps for each example
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=device).long()

        # Sample noise
        noise = torch.randn_like(actions)

        # Add noise to actions (forward diffusion)
        # x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
        noisy_actions = self.q_sample(actions, t, noise)

        # Predict noise using the denoising network: ε_θ(x_t, s, t)
        noise_pred = self.denoiser(noisy_actions, states, t)

        # Training objective (simplified DDPM loss):
        # L_simple = E_{t, x_0, ε}[||ε - ε_θ(x_t, t)||²]
        #
        # We train the network to predict the noise that was added,
        # which is equivalent to predicting the score function ∇log p(x_t)
        # and allows us to iteratively denoise during sampling.
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, states, n_samples=1):
        """
        Generate actions using reverse diffusion process.

        This implements the DDPM sampling algorithm:
        1. Start with x_T ~ N(0, I) (pure noise)
        2. For t = T, T-1, ..., 1:
           - Predict noise: ε_θ(x_t, s, t)
           - Compute mean: μ_θ = (1/√α_t)·(x_t - (β_t/√(1-ᾱ_t))·ε_θ)
           - Sample: x_{t-1} = μ_θ + σ_t·z, where z ~ N(0, I)
        3. Return x_0 (clean action)

        This reverses the forward diffusion process, starting from noise
        and iteratively denoising to produce a clean action conditioned on state.

        Args:
            states: (B, state_dim) - state observations
            n_samples: number of action samples to generate per state
        Returns:
            actions: (B, n_samples, action_dim) - generated actions
        """
        B = states.shape[0]
        device = states.device

        # Start from random noise
        actions = torch.randn(B, n_samples, self.action_dim, device=device)

        # Iteratively denoise
        for t in reversed(range(self.n_diffusion_steps)):
            # Prepare timestep tensor
            t_tensor = torch.full((B * n_samples,), t, device=device, dtype=torch.long)

            # Flatten for batch processing
            actions_flat = actions.view(B * n_samples, self.action_dim)
            states_expanded = states.unsqueeze(1).repeat(1, n_samples, 1).view(B * n_samples, self.state_dim)

            # Predict noise: ε_θ(x_t, s, t) where s is the state observation
            noise_pred = self.denoiser(actions_flat, states_expanded, t_tensor)

            # Compute denoising update
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            # DDPM reverse process equation:
            # μ_θ(x_t, t) = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t))
            # This is the predicted mean of p(x_{t-1} | x_t)
            #
            # Derivation: We want to predict x_0 from x_t, then use it to estimate x_{t-1}
            # From q(x_t | x_0), we can rearrange to estimate:
            # x_0 ≈ (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
            # Then use this to compute the mean of the reverse distribution
            actions_flat = (1.0 / torch.sqrt(alpha_t)) * (
                actions_flat - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * noise_pred
            )

            # Add noise (except at last step)
            # Full reverse step: x_{t-1} = μ_θ(x_t, t) + σ_t · z, where z ~ N(0, I)
            # σ_t = √β_t (simplified version; could also use posterior variance)
            if t > 0:
                noise = torch.randn_like(actions_flat)
                sigma_t = torch.sqrt(beta_t)
                actions_flat = actions_flat + sigma_t * noise

            # Reshape back
            actions = actions_flat.view(B, n_samples, self.action_dim)

        return actions


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

            for t in range(self.episode_length):
                ee_pos = self._forward_kinematics(joint_angles)

                # State: [joint_angles, ee_pos, goal_pos]
                state = np.concatenate([joint_angles, ee_pos, goal_pos])

                # Expert action
                action = self._expert_policy(joint_angles, ee_pos, goal_pos)

                all_states.append(state)
                all_actions.append(action)

                # Update joint angles
                joint_angles = joint_angles + action * 0.1  # dt = 0.1
                joint_angles = np.clip(joint_angles, -np.pi, np.pi)

        return np.array(all_states), np.array(all_actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.FloatTensor(self.states[idx])
        action = torch.FloatTensor(self.actions[idx])
        return state, action


def train(model, train_loader, test_loader, optimizer, epochs, device):
    """Train the diffusion policy."""
    for epoch in range(epochs):
        model.train()
        training_loss = 0

        for states, actions in tqdm(train_loader, desc=f"Epoch {epoch}"):
            states, actions = states.to(device), actions.to(device)

            optimizer.zero_grad()

            # Compute diffusion loss
            loss = model(actions, states)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
                loss = model(actions, states)
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

        for t in range(20):
            ee_pos = dataset._forward_kinematics(joint_angles)
            state = np.concatenate([joint_angles, ee_pos, goal_pos])

            # Get action from diffusion policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                # Sample actions (can generate multiple and pick best)
                sampled_actions = model.sample(state_tensor, n_samples=1)
                action = sampled_actions[0, 0, :].cpu().numpy()

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
    hidden_dim = 256
    n_diffusion_steps = 100
    batch_size = 256
    epochs = 30
    lr = 3e-4

    # Create datasets
    print("Generating expert demonstrations...")
    train_dataset = ExpertDemonstrationDataset(n_episodes=5000, episode_length=20)
    test_dataset = ExpertDemonstrationDataset(n_episodes=500, episode_length=20)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = RobotPolicyDiffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        n_diffusion_steps=n_diffusion_steps,
        hidden_dim=hidden_dim
    )
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train
    print("\nTraining...")
    model = train(model, train_loader, test_loader, optimizer, epochs, device)

    # Evaluate
    print("\nEvaluating learned policy...")
    evaluate_policy(model, device, n_episodes=20)
