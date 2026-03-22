"""
Variational Autoencoder (VAE) for 64x64 RGB images.

Reference: https://lilianweng.github.io/posts/2018-08-12-vae/

The VAE learns a generative model p(x|z) and an approximate posterior q(z|x)
by maximizing the Evidence Lower Bound (ELBO):

    ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

where:
  - q(z|x) = N(mu(x), sigma^2(x))  (encoder)
  - p(z) = N(0, I)                   (prior)
  - p(x|z) is parameterized by the decoder
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Maps input x to latent distribution parameters (mu, log_var)."""

    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Maps latent z back to image space."""

    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Tanh(),  # output in [-1, 1] to match normalized input
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def sample(self, n, device):
        """Sample from the prior p(z) = N(0, I) and decode."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    @staticmethod
    def loss(x, x_recon, mu, log_var):
        """
        ELBO loss = reconstruction loss + KL divergence.

        Reconstruction: MSE between input and output.
        KL: closed-form for two Gaussians:
            KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return recon_loss, kl_loss
