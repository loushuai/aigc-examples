# Denoising Diffusion Probabilistic Models (DDPM)

## Forward Process
The forward process gradually adds Gaussian noise to the data over $`T`$ time steps. Given a data point $`\mathbf{x}_0 \sim q(\mathbf{x}_0)`$, the forward process produces a sequence of increasingly noisy samples $`\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T`$:
```math
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I})
```
where $`\beta_1, \beta_2, \dots, \beta_T`$ is a variance schedule. In this implementation we use a linear schedule from $`\beta_1 = 10^{-4}`$ to $`\beta_T = 0.02`$ with $`T = 1000`$.

A key property is that we can sample $`\mathbf{x}_t`$ directly from $`\mathbf{x}_0`$ in closed form. Defining $`\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)`$:
```math
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I})
```
which can be reparameterized as:
```math
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
```

## Reverse Process
The reverse process learns to denoise, starting from $`\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})`$ and iteratively removing noise:
```math
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\!\Big(\mathbf{x}_{t-1};\, \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\, \beta_t \mathbf{I}\Big)
```

The mean $`\boldsymbol{\mu}_\theta`$ is parameterized via a noise prediction network $`\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)`$:
```math
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{1-\beta_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)
```

## Training
The training objective simplifies to predicting the noise $`\boldsymbol{\epsilon}`$ that was added during the forward process [[1]][#reference-1]:
```math
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\,\mathbf{x}_0,\,\boldsymbol{\epsilon}} \Big[ \big\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big\|^2 \Big]
```

### Algorithm: Training
1. **repeat**
2. $`\quad \mathbf{x}_0 \sim q(\mathbf{x}_0)`$
3. $`\quad t \sim \text{Uniform}(\{1, \dots, T\})`$
4. $`\quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})`$
5. $`\quad`$ Take gradient descent step on $`\nabla_\theta \big\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\big(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\, t\big) \big\|^2`$
6. **until** converged

## Inference
### Algorithm: Sampling
1. $`\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})`$
2. **for** $`t = T, T-1, \dots, 1`$ **do**
3. $`\quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})`$ if $`t > 1`$, else $`\mathbf{z} = 0`$
4. $`\quad \mathbf{x}_{t-1} = \frac{1}{\sqrt{1-\beta_t}} \Big( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big) + \sqrt{\beta_t}\,\mathbf{z}`$
5. **end for**
6. **return** $`\mathbf{x}_0`$

## Reference
[1] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

[#reference-1]: https://arxiv.org/abs/2006.11239
