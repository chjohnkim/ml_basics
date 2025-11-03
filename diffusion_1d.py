import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# Simple MLP to predict noise
class NoisePredictor(nn.Module):
    def __init__(self, T):
        super().__init__()
        # Diffusion parameters
        self.T = T
        # Variance schedule: β_t controls noise added at each step
        self.beta = torch.linspace(1e-4, 0.02, self.T)  # β_t ∈ (0, 1)
        # α_t = 1 - β_t (retention factor at step t)
        self.alpha = 1 - self.beta
        
        # ᾱ_t = ∏_{s=1}^t α_s (cumulative product for direct sampling from x_0)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Noise prediction model: ε_θ(x_t, t) 
        # Predicts the noise that was added to get from x_0 to x_t
        self.net = nn.Sequential(
            nn.Linear(2, 64),  # input: [noisy_x, timestep]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # output: predicted noise
        )
    
    def forward(self, x, t):
        """Neural network ε_θ(x_t, t) that predicts noise"""
        # Ensure both x and t are [batch, 1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float() / self.T  # Normalize timestep
        output = self.net(torch.cat([x, t], dim=-1))
        return output.squeeze(-1)

    def forward_diffusion(self, x0, t):
        """
        Forward process q(x_t | x_0): Add noise to clean data
        
        Key equation (closed form sampling):
        q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)
        
        Reparameterization: x_t = √ᾱ_t x_0 + √(1 - ᾱ_t) ε, where ε ~ N(0, I)
        """
        # Sample noise: ε ~ N(0, I)
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t]
        # x_t = √ᾱ_t x_0 + √(1 - ᾱ_t) ε
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise
    
    @torch.no_grad()
    def backward_diffusion(self, n_samples, use_ddim=False, ddim_steps=None, eta=0.0):
        """
        Reverse process p_θ(x_{t-1} | x_t): Iteratively denoise
        
        DDPM (use_ddim=False):
            Key equation (reverse diffusion):
            p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)        
            Where mean is computed as:
            μ_θ(x_t, t) = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t))
            And variance: σ_t² = β_t

        DDIM (use_ddim=True):
            Deterministic sampling using predicted x_0
            x_{t-1} = √ᾱ_{t-1} x̂_0 + √(1 - ᾱ_{t-1} - σ_t²) ε_θ(x_t, t) + σ_t ε
            where x̂_0 = (x_t - √(1-ᾱ_t) ε_θ(x_t, t)) / √ᾱ_t

        """
        # Start from pure noise: x_T ~ N(0, I)
        x = torch.randn(n_samples)

        if use_ddim and ddim_steps is not None:
            # Create subsequence of timesteps (e.g., [999, 949, 899, ...] for 20 steps out of 1000)
            # Uniform spacing from T-1 to 0
            step_size = self.T // ddim_steps
            timesteps = list(range(0, self.T, step_size))
            timesteps = timesteps[:ddim_steps]  # Ensure exactly ddim_steps
            timesteps = list(reversed(timesteps))  # Go from T to 0
        else:
            # Use all timesteps
            timesteps = list(reversed(range(self.T)))

        # Iteratively denoise from t=T to t=1
        for i, t_idx in enumerate(timesteps):
            t = torch.full((n_samples,), t_idx, dtype=torch.long)

            # Predict noise: ε_θ(x_t, t)
            predicted_noise = self.forward(x, t)

            # Get parameters for this timestep
            alpha_bar_t = self.alpha_bar[t_idx]  # ᾱ_t

            if use_ddim:
                # DDIM: Deterministic denoising
                # Predict x_0: x̂_0 = (x_t - √(1-ᾱ_t) ε_θ) / √ᾱ_t
                pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                
                # Get previous timestep index
                if i < len(timesteps) - 1:
                    t_prev_idx = timesteps[i + 1]
                    alpha_bar_prev = self.alpha_bar[t_prev_idx]  # ᾱ_{t-1}
                    
                    # Compute variance: σ_t² = η² · β̃_t where β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t
                    # For skipped timesteps, this generalizes naturally
                    sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                    
                    # Direction pointing to x_t
                    dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * predicted_noise
                    
                    # DDIM update: x_{t-1} = √ᾱ_{t-1} x̂_0 + direction + noise
                    x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
                    
                    if eta > 0:  # Add stochasticity if eta > 0
                        noise = torch.randn_like(x)
                        x = x + sigma_t * noise
                else:
                    # Final step: just return predicted x_0
                    x = pred_x0

            else:
                # DDPM: Stochastic denoising
                alpha_t = self.alpha[t_idx]      # α_t
                beta_t = self.beta[t_idx]        # β_t

                # Compute mean: μ_θ(x_t, t) = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t))
                x = (x - beta_t / torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)
                
                # Add noise (except at last step): x_{t-1} = μ_θ + σ_t z, where z ~ N(0, I)
                # This gives us: x_{t-1} ~ N(μ_θ, σ_t²I)
                if t_idx > 0:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise  # σ_t = √β_t

        return x
    
# Create bimodal dataset
def sample_data(n):
    mode = torch.rand(n) > 0.5
    return torch.randn(n) * 0.5 + mode * 4 - 2


def train_diffusion(model, optimizer, T, num_epochs, batch_size):
    for step in range(num_epochs):
        x0 = sample_data(batch_size)
        t = torch.randint(0, T, (batch_size,))
        xt, noise = model.forward_diffusion(x0, t)
        predicted_noise = model(xt, t)
        
        loss = nn.MSELoss()(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")


def main():
    # Hyperparameters:
    # Diffusion
    T = 1000
    # Training
    num_epochs = 5000
    batch_size = 128
    learning_rate = 1e-3
    use_ddim = True
    ddim_steps = 10

    # Training
    model = NoisePredictor(T)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_diffusion(model, optimizer, T, num_epochs, batch_size)

    # Generate and plot
    t0 = time.time()
    generated = model.backward_diffusion(1000, use_ddim=use_ddim, ddim_steps=ddim_steps)
    print(f"Inference time {'WITH' if use_ddim else 'WITHOUT'} DDIM: {time.time()-t0}")
    original = sample_data(1000)

    plt.figure(figsize=(10, 4))
    plt.hist(original.numpy(), bins=50, alpha=0.5, label='Original')
    plt.hist(generated.numpy(), bins=50, alpha=0.5, label='Generated')
    plt.legend()
    plt.title('1D Diffusion Model')
    plt.show()

if __name__=="__main__":
    main()