# Variational Autoencoder (VAE)

## Overview
A VAE learns a generative model by encoding data into a structured latent space and decoding samples back to data space. Unlike a regular autoencoder which maps inputs to fixed points, a VAE maps inputs to probability distributions, producing a smooth latent space that supports sampling and interpolation.

## ELBO
We want to maximize $`\log p(\mathbf{x})`$, but this is intractable. Instead we maximize the **Evidence Lower Bound (ELBO)**:
```math
\log p(\mathbf{x}) \geq \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction}} - \underbrace{D_{KL}\big(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})\big)}_{\text{KL divergence}} = \text{ELBO}
```
where:
- $`q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}),\, \boldsymbol{\sigma}_\phi^2(\mathbf{x}))`$ is the encoder (approximate posterior)
- $`p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})`$ is the prior
- $`p_\theta(\mathbf{x}|\mathbf{z})`$ is the decoder

## Reparameterization Trick
Sampling $`\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})`$ is not differentiable. The reparameterization trick resolves this by writing:
```math
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
```
Now $`\mathbf{z}`$ is a deterministic function of $`\boldsymbol{\mu}`$, $`\boldsymbol{\sigma}`$, and $`\boldsymbol{\epsilon}`$, allowing gradients to flow through the encoder.

## Training
The loss to minimize is:
```math
\mathcal{L}(\theta, \phi) = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{MSE reconstruction}} + \beta \cdot \underbrace{D_{KL}\big(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})\big)}_{\text{KL regularizer}}
```
The KL divergence has a closed-form solution for two Gaussians:
```math
D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
```
Setting $`\beta = 1`$ gives the standard VAE; $`\beta > 1`$ gives the **beta-VAE** which encourages stronger disentanglement at the cost of reconstruction quality.

### Algorithm: Training
1. **repeat**
2. $`\quad \mathbf{x} \sim p_{\text{data}}(\mathbf{x})`$
3. $`\quad \boldsymbol{\mu}, \log \boldsymbol{\sigma}^2 \leftarrow \text{Encoder}_\phi(\mathbf{x})`$
4. $`\quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})`$
5. $`\quad \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}`$
6. $`\quad \hat{\mathbf{x}} \leftarrow \text{Decoder}_\theta(\mathbf{z})`$
7. $`\quad`$ Take gradient descent step on $`\|\mathbf{x} - \hat{\mathbf{x}}\|^2 - \frac{\beta}{2} \sum_j (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)`$
8. **until** converged

## Inference
### Algorithm: Sampling
1. $`\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})`$
2. $`\mathbf{x} = \text{Decoder}_\theta(\mathbf{z})`$
3. **return** $`\mathbf{x}`$

## Architecture
The encoder uses 4 stride-2 convolutional layers to downsample 64x64 images to a 4x4 feature map, then projects to $`\boldsymbol{\mu}`$ and $`\log \boldsymbol{\sigma}^2`$ (each 128-dim). The decoder mirrors this with transposed convolutions and a final Tanh activation.

```
Encoder: [3,64,64] -> Conv(32) -> Conv(64) -> Conv(128) -> Conv(256) -> [256,4,4] -> FC -> mu, log_var
Decoder: z -> FC -> [256,4,4] -> ConvT(128) -> ConvT(64) -> ConvT(32) -> ConvT(3) -> Tanh -> [3,64,64]
```

## Reference
[1] [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

[#reference-1]: https://arxiv.org/abs/1312.6114

[2] [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

[#reference-2]: https://openreview.net/forum?id=Sy2fzU9gl
