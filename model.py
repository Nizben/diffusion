import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

###############################################
# 1. Diffusion Process and Noise Schedule
###############################################

class Diffusion:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device='cuda'):
        """
        timesteps: Number of diffusion steps.
        beta_start, beta_end: Defines a linear schedule for betas.
        device: 'cuda' or 'cpu'.
        """
        self.timesteps = timesteps
        # Create a linear schedule for beta (noise variance) values.
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)  # shape: (T,)
        self.alpha = 1.0 - self.beta                                      # shape: (T,)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)                 # shape: (T,)
        self.device = device

    def forward_diffusion_sample(self, x0, t):
        """
        Takes an original image x0 and a timestep t and returns a noised image x_t.
        
        x0: tensor of shape (B, C, H, W) — original images.
        t: tensor of shape (B,) — timesteps (integers between 0 and T-1) for each image.
        
        Returns:
            x_t: Noised image tensor, same shape as x0.
            noise: The noise that was added, also (B, C, H, W).
        
        The formula implemented is:
          x_t = sqrt(alpha_bar_t)*x0 + sqrt(1 - alpha_bar_t)*noise
        """
        # Extract the corresponding cumulative product of alphas (alpha_bar) for each t.
        # self.alpha_bar[t] returns a tensor of shape (B,)
        sqrt_alpha_bar = self.alpha_bar[t].sqrt()         # shape: (B,)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt()  # shape: (B,)
        # Reshape to (B, 1, 1, 1) so we can broadcast multiplication over (C, H, W)
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)
        # Sample noise from a standard normal distribution.
        noise = torch.randn_like(x0)  # shape: (B, C, H, W)
        # Create the noised image
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

###############################################
# 2. Time Embedding and the Denoising Network
###############################################

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal embeddings for the timesteps.
    
    timesteps: Tensor of shape (N,) containing timesteps.
    embedding_dim: Dimension of the time embedding.
    
    Returns:
        embeddings: Tensor of shape (N, embedding_dim)
    """
    half_dim = embedding_dim // 2
    # Use logarithmic scaling
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb_scale)
    emb = timesteps.float()[:, None] * emb[None, :]  # shape: (N, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # shape: (N, embedding_dim)
    if embedding_dim % 2 == 1:  # if embedding_dim is odd
        emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1, device=timesteps.device)], dim=1)
    return emb

class SimpleDenoiser(nn.Module):
    def __init__(self, img_channels, base_channels=64, time_embedding_dim=128):
        """
        A minimal CNN that predicts the noise added to an image at timestep t.
        
        img_channels: Number of image channels (e.g. 3 for RGB).
        base_channels: Number of channels in the hidden layers.
        time_embedding_dim: Dimensionality for time embedding.
        """
        super(SimpleDenoiser, self).__init__()
        # Create a small MLP to embed the timestep.
        self.time_embed = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, base_channels)
        )
        # A few convolutional layers.
        self.conv1 = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        """
        x: Noised image tensor, shape (B, C, H, W)
        t: Timestep tensor, shape (B,) containing integers.
        
        Returns:
            Predicted noise tensor of shape (B, C, H, W).
        """
        # Get sinusoidal time embeddings (shape: (B, time_embedding_dim))
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        # Process time embedding through the MLP to get a vector of size (B, base_channels)
        t_emb = self.time_embed(t_emb)  # shape: (B, base_channels)
        # Reshape to (B, base_channels, 1, 1) for broadcasting.
        t_emb = t_emb[:, :, None, None]

        # Process the image through convolutions.
        h = self.conv1(x)             # shape: (B, base_channels, H, W)
        h = h + t_emb                 # add time conditioning to each feature map
        h = self.relu(h)
        h = self.conv2(h)             # shape remains (B, base_channels, H, W)
        h = self.relu(h)
        h = self.conv3(h)             # shape: (B, img_channels, H, W)
        return h

###############################################
# 3. Training Script with Data Preprocessing
###############################################

def train_diffusion(model, diffusion, dataloader, optimizer, epochs, device):
    """
    Trains the diffusion model.
    
    model: The denoising network.
    diffusion: Instance of the Diffusion class.
    dataloader: PyTorch DataLoader yielding training images.
    optimizer: Optimizer (e.g. Adam).
    epochs: Number of training epochs.
    device: 'cuda' or 'cpu'.
    """
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            # images: (B, C, H, W) where B=batch size.
            images = images.to(device)
            B = images.size(0)
            # Sample a random timestep for each image in the batch (uniformly from 0 to T-1).
            t = torch.randint(0, diffusion.timesteps, (B,), device=device).long()  # shape: (B,)
            # Get the noised image x_t and the noise that was added.
            x_t, noise = diffusion.forward_diffusion_sample(images, t)  # both are (B, C, H, W)
            # The network tries to predict the noise added.
            predicted_noise = model(x_t, t)  # output shape: (B, C, H, W)
            loss = mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")

###############################################
# 4. Sampling (Generating New Images)
###############################################

@torch.no_grad()
def sample_image(model, diffusion, shape, device):
    """
    Reverse diffusion: starting from pure noise, gradually denoise to generate an image.
    
    model: The trained denoising network.
    diffusion: The Diffusion scheduler.
    shape: Tuple (B, C, H, W) — desired shape of generated images.
    device: 'cuda' or 'cpu'.
    
    Returns:
        Generated images of shape (B, C, H, W)
    """
    # Start with pure Gaussian noise.
    x = torch.randn(shape, device=device)
    # Iterate backwards from t = T-1 down to 0.
    for t in reversed(range(diffusion.timesteps)):
        # Create a tensor of shape (B,) filled with the current timestep.
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        # Predict the noise at this step.
        predicted_noise = model(x, t_batch)
        # Retrieve the diffusion hyperparameters for timestep t.
        beta_t = diffusion.beta[t]
        alpha_t = diffusion.alpha[t]
        alpha_bar_t = diffusion.alpha_bar[t]
        # Compute the reverse process mean following the DDPM paper:
        #   x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t/sqrt(1 - alpha_bar_t))*predicted_noise)
        # For t > 0, add a small amount of noise; for t == 0, no noise is added.
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        coef1 = 1 / (alpha_t**0.5)
        coef2 = (beta_t / ((1 - alpha_bar_t)**0.5))
        x = coef1 * (x - coef2 * predicted_noise) + (beta_t**0.5) * noise
    return x

###############################################
# 5. Main: Data Loading, Training, and Sampling
###############################################

def main():
    # Set up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters.
    epochs = 5             # For a demo; increase for better results.
    batch_size = 64
    learning_rate = 1e-3
    img_size = 32          # CIFAR-10 images are 32x32.
    channels = 3           # RGB images.
    timesteps = 100        # Number of diffusion steps.

    # Data Preprocessing: Define image transformations.
    transform = transforms.Compose([
        transforms.Resize(img_size),         # Ensure images are img_size x img_size.
        transforms.ToTensor(),               # Convert PIL image to tensor (C x H x W) in [0,1].
        transforms.Normalize((0.5, 0.5, 0.5),  # Normalize to [-1, 1] (helps training).
                             (0.5, 0.5, 0.5))
    ])
    # Load CIFAR-10 dataset (or swap in your own dataset of real images).
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 transform=transform, download=True)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the diffusion scheduler.
    diffusion = Diffusion(timesteps=timesteps, device=device)
    # Initialize the denoising model.
    model = SimpleDenoiser(img_channels=channels, base_channels=64, time_embedding_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model.
    train_diffusion(model, diffusion, dataloader, optimizer, epochs, device)

    # After training, generate some samples.
    sample = sample_image(model, diffusion, (16, channels, img_size, img_size), device)
    # Unnormalize the generated images (from [-1,1] back to [0,1]) for visualization.
    sample = (sample + 1) / 2

    # Visualize the generated samples.
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils
    grid = vutils.make_grid(sample, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()

if __name__ == '__main__':
    main()

