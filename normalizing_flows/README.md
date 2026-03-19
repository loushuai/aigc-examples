# Normalizing Flows

Normalizing flows are a family of generative models that construct complex distributions by transforming a simple base distribution (e.g., Gaussian) through a sequence of invertible and differentiable mappings. Unlike other generative models, normalizing flows allow exact likelihood computation and efficient sampling.

## Change of Variables

Given an invertible transformation $`\mathbf{z} = f(\mathbf{x})`$ where $`f`$ maps data $`\mathbf{x}`$ to a latent variable $`\mathbf{z}`$, the change of variables formula gives us:
```math
p(\mathbf{x}) = p(\mathbf{z}) \left| \det \frac{\partial f}{\partial \mathbf{x}} \right|
```
Taking the logarithm:
```math
\log p(\mathbf{x}) = \log p(\mathbf{z}) + \log \left| \det \frac{\partial f}{\partial \mathbf{x}} \right|
```

For a composition of $`K`$ transformations $`f = f_K \circ f_{K-1} \circ \cdots \circ f_1`$:
```math
\log p(\mathbf{x}) = \log p(\mathbf{z}_K) + \sum_{i=1}^{K} \log \left| \det \frac{\partial f_i}{\partial \mathbf{z}_{i-1}} \right|
```
where $`\mathbf{z}_0 = \mathbf{x}`$ and $`\mathbf{z}_K = \mathbf{z}`$.

## Affine Coupling Layer

The affine coupling layer [[1]][#reference-1] is the core building block. It splits the input into two parts and transforms one part conditioned on the other.

**Forward transformation:** split $`\mathbf{x}`$ into $`\mathbf{x}_1`$ and $`\mathbf{x}_2`$ along the channel dimension:
```math
\mathbf{y}_1 = \mathbf{x}_1
```
```math
\mathbf{y}_2 = \mathbf{x}_2 \odot \exp(\mathbf{s}(\mathbf{x}_1)) + \mathbf{t}(\mathbf{x}_1)
```
where $`\mathbf{s}(\cdot)`$ and $`\mathbf{t}(\cdot)`$ are neural networks that output the scale and translation parameters.

**Inverse transformation** (trivially computed):
```math
\mathbf{x}_1 = \mathbf{y}_1
```
```math
\mathbf{x}_2 = (\mathbf{y}_2 - \mathbf{t}(\mathbf{y}_1)) \odot \exp(-\mathbf{s}(\mathbf{y}_1))
```

**Log-determinant of the Jacobian** (efficient to compute since the Jacobian is triangular):
```math
\log \left| \det \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right| = \sum_j s(\mathbf{x}_1)_j
```
Note that neither $`\mathbf{s}`$ nor $`\mathbf{t}`$ need to be invertible — only the coupling layer as a whole must be invertible. This allows using arbitrarily complex neural networks for $`\mathbf{s}`$ and $`\mathbf{t}`$.

## Activation Normalization (ActNorm)

ActNorm [[2]][#reference-2] performs a learnable per-channel affine transformation:
```math
\mathbf{y} = \mathbf{s} \odot \mathbf{x} + \mathbf{b}
```
The parameters $`\mathbf{s}`$ and $`\mathbf{b}`$ are initialized using data-dependent initialization so that the first mini-batch has zero mean and unit variance per channel. The log-determinant is:
```math
\log \left| \det \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right| = H \cdot W \cdot \sum_c \log |s_c|
```

## Invertible 1x1 Convolution

An invertible 1×1 convolution [[2]][#reference-2] generalizes the permutation of channels. It is parameterized by a weight matrix $`\mathbf{W} \in \mathbb{R}^{C \times C}`$:
```math
\mathbf{y}_{h,w} = \mathbf{W} \, \mathbf{x}_{h,w}
```
with log-determinant:
```math
\log \left| \det \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right| = H \cdot W \cdot \log |\det \mathbf{W}|
```
Using LU decomposition $`\mathbf{W} = \mathbf{P}\mathbf{L}\mathbf{U}`$ makes both the forward pass and log-determinant computation efficient ($`O(C)`$ instead of $`O(C^3)`$).

## Multi-Scale Architecture

Following RealNVP [[1]][#reference-1] and Glow [[2]][#reference-2], we use a multi-scale architecture with squeeze operations. At each level:

1. **Squeeze**: Reshape spatial dimensions into channels: $`(C, H, W) \rightarrow (4C, H/2, W/2)`$
2. **Flow Steps**: Apply $`K`$ flow steps, each consisting of ActNorm → Inv 1×1 Conv → Affine Coupling
3. **Split**: Factor out half the channels to the latent space (except at the last level)

This multi-scale design improves computational efficiency and allows the model to capture features at different spatial resolutions.

## Training

The training objective is to maximize the log-likelihood of the data, or equivalently minimize the negative log-likelihood:
```math
\mathcal{L} = -\frac{1}{|D|} \sum_{\mathbf{x} \in D} \log p(\mathbf{x}) = -\frac{1}{|D|} \sum_{\mathbf{x} \in D} \left[ \log p(\mathbf{z}) + \log \left| \det \frac{\partial f}{\partial \mathbf{x}} \right| \right]
```
where $`p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})`$ is a standard Gaussian. This is a direct maximum likelihood objective — unlike VAEs (which optimize an ELBO) or GANs (which use an adversarial loss).

#### Algorithm 1: Normalizing Flow Training

---

**Require:** A dataset of samples $`\mathbf{x} \sim p_{\text{data}}`$, flow model $`f_\theta`$ with base distribution $`p(\mathbf{z}) = \mathcal{N}(0, I)`$

1. **for** each mini-batch of data **do**
2. &nbsp;&nbsp;&nbsp;&nbsp;Sample a data example $`\mathbf{x}`$ from the dataset.
3. &nbsp;&nbsp;&nbsp;&nbsp;Compute $`\mathbf{z}, \log|\det J|`$ = $`f_\theta(\mathbf{x})`$ (forward pass through all flow layers).
4. &nbsp;&nbsp;&nbsp;&nbsp;Compute loss:

```math
\mathcal{L}(\theta) = -\left[ \log p(\mathbf{z}) + \log \left| \det J \right| \right] = -\left[ -\frac{1}{2}\|\mathbf{z}\|^2 - \frac{d}{2}\log(2\pi) + \log \left| \det J \right| \right]
```

5. &nbsp;&nbsp;&nbsp;&nbsp;Update $`\theta`$ via gradient descent.
6. **end for**

## Inference

After training, generating new samples is straightforward — sample from the base distribution and run the inverse of the flow:

#### Algorithm 2: Sampling from a Normalizing Flow

---

**Require:** Trained flow model $`f_\theta`$, temperature $`T`$

1. Sample $`\mathbf{z} \sim \mathcal{N}(0, T^2 \mathbf{I})`$
2. Compute $`\mathbf{x} = f_\theta^{-1}(\mathbf{z})`$ (inverse pass through all flow layers in reverse order)
3. **return** $`\mathbf{x}`$

---

The temperature $`T`$ controls the diversity of the generated samples. Lower temperatures ($`T < 1`$) produce sharper but less diverse samples, while $`T = 1`$ samples from the exact learned distribution.


## Reference
[1] [Density estimation using Real-NVP](https://arxiv.org/abs/1605.08803)

[#reference-1]: https://arxiv.org/abs/1605.08803

[2] [Glow: Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039)

[#reference-2]: https://arxiv.org/abs/1807.03039

[3] [Flow-based Deep Generative Models (Lilian Weng)](https://lilianweng.github.io/posts/2018-10-13-flow-models/)

[#reference-3]: https://lilianweng.github.io/posts/2018-10-13-flow-models/
