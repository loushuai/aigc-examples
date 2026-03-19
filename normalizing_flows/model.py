import torch
import torch.nn as nn
import numpy as np

from normalizing_flows.modules import (
    ActNorm, Invertible1x1Conv, AffineCouplingLayer, SplitPrior, squeeze, unsqueeze
)


class FlowStep(nn.Module):
    """A single step in the normalizing flow: ActNorm -> Inv1x1Conv -> AffineCoupling."""

    def __init__(self, num_channels, hidden_channels=256):
        super().__init__()
        self.actnorm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.coupling = AffineCouplingLayer(num_channels, hidden_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        log_det_total = torch.zeros(batch_size, device=x.device)
        x, log_det = self.actnorm(x)
        log_det_total += log_det
        x, log_det = self.inv_conv(x)
        log_det_total += log_det
        x, log_det = self.coupling(x)
        log_det_total += log_det
        return x, log_det_total

    def inverse(self, z):
        z = self.coupling.inverse(z)
        z = self.inv_conv.inverse(z)
        z = self.actnorm.inverse(z)
        return z


class FlowLevel(nn.Module):
    """A level in the multi-scale architecture: Squeeze -> K FlowSteps -> Split.

    When split=True, uses a learned split prior (matching TF split2d_prior)
    to model the factored-out channels conditioned on the kept channels.
    """

    def __init__(self, num_channels, num_steps, hidden_channels=256, split=True):
        super().__init__()
        self.steps = nn.ModuleList([
            FlowStep(num_channels, hidden_channels) for _ in range(num_steps)
        ])
        self.split = split
        if split:
            self.split_prior = SplitPrior(num_channels // 2)

    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        x = squeeze(x)
        for step in self.steps:
            x, log_det = step(x)
            log_det_total += log_det
        if self.split:
            x, z_out = x.chunk(2, dim=1)
            # Learned prior: log p(z_out | x)
            log_p_split = self.split_prior.log_prob(x, z_out)
            log_det_total += log_p_split
        else:
            z_out = x
            x = None
        return x, z_out, log_det_total

    def inverse(self, x, z_out):
        if self.split:
            z = torch.cat([x, z_out], dim=1)
        else:
            z = z_out
        for step in reversed(self.steps):
            z = step.inverse(z)
        z = unsqueeze(z)
        return z

    def sample_split(self, x, temperature=1.0):
        """Sample z_out from the learned split prior given x."""
        return self.split_prior.sample(x, temperature)


class GlowModel(nn.Module):
    """Glow: Generative Flow with Invertible 1x1 Convolutions.

    Multi-scale normalizing flow architecture for image generation.
    At each scale level: Squeeze -> K flow steps -> Split (factor out half channels).

    Args:
        in_channels: Number of input channels (3 for RGB).
        num_levels: Number of multi-scale levels.
        num_steps: Number of flow steps per level.
        hidden_channels: Hidden channels in coupling networks.
    """

    def __init__(self, in_channels=3, num_levels=3, num_steps=8, hidden_channels=256):
        super().__init__()
        self.num_levels = num_levels
        self.levels = nn.ModuleList()

        c = in_channels
        for i in range(num_levels):
            c = c * 4  # After squeeze
            is_last = (i == num_levels - 1)
            self.levels.append(
                FlowLevel(c, num_steps, hidden_channels, split=not is_last)
            )
            if not is_last:
                c = c // 2  # After split

    def forward(self, x):
        """Forward pass: data -> latent.

        Returns:
            z_list: List of latent tensors at each scale.
            log_det: Total log-determinant of the Jacobian.
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z_list = []
        h = x
        for level in self.levels:
            h, z_out, log_det = level(h)
            log_det_total += log_det
            z_list.append(z_out)
        return z_list, log_det_total

    def inverse(self, z_list):
        """Inverse pass: latent -> data."""
        h = None
        for i in reversed(range(self.num_levels)):
            h = self.levels[i].inverse(h, z_list[i])
        return h

    def log_prob(self, x):
        """Compute log p(x) using change of variables.

        log p(x) = log p(z_final) + sum(log p(z_split_i | h_i)) + log |det(df/dx)|
                   - D * log(n_bins)

        The split prior log-probs are already included in log_det by FlowLevel.
        Only the final-level z uses a standard Gaussian prior.
        The dequantization correction (- D * log(256)) converts from continuous
        density to discrete bits, matching the TF reference.
        """
        z_list, log_det = self.forward(x)
        # Standard Gaussian prior only on the last z (full latent, no split)
        z_final = z_list[-1]
        log_pz = -0.5 * (z_final ** 2 + np.log(2 * np.pi)).sum(dim=[1, 2, 3])
        # Dequantization correction: -D * log(n_bins), n_bins=256
        n_pixels = x.shape[1] * x.shape[2] * x.shape[3]
        log_dequant = -np.log(256.0) * n_pixels
        return log_pz + log_det + log_dequant

    def sample(self, num_samples, device, temperature=0.7):
        """Generate samples by inverting the flow.

        For the final level, sample z ~ N(0, T^2 I).
        For intermediate levels, sample z_out from the learned split prior.
        """
        # First, sample the final-level latent
        c = 3
        h, w = 64, 64
        for i in range(self.num_levels):
            c = c * 4
            h, w = h // 2, w // 2
            if i < self.num_levels - 1:
                c = c // 2

        z_final = torch.randn(num_samples, c, h, w, device=device) * temperature

        # Reverse through levels
        x = None
        for i in reversed(range(self.num_levels)):
            if i == self.num_levels - 1:
                z_out = z_final
            else:
                # Sample z_out from learned split prior conditioned on x
                z_out = self.levels[i].sample_split(x, temperature)
            x = self.levels[i].inverse(x, z_out)
        return x
