# Score Matching

## Langevine Dynamics
The core of score matching is Langevine dynamics:
```math
\tilde{\mathbf{x}}_{t} = \tilde{\mathbf{x}}_{t-1} + \frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p(\tilde{\mathbf{x}}_{t-1}) + \sqrt{\epsilon} \mathbf{z}_{t}
```
where $`\mathbf{z}_t \sim \mathcal{N}(0, I)`$. The gradient of the logarithm of a distribution is defined as the score.

Given a fixed step size $`\epsilon>0`$ and an initial value $`\tilde{\mathbf{x}}_0\sim\pi(\mathbf{x})`$ with $`\pi`$ being an prior distribution, we can use this process to produce samples from a probability density $`p(x)`$ using only the score function $`\nabla_{\mathbf{x}} \log p(\tilde{\mathbf{x}}_{t-1})`$.\
You can find an intuitve introduction of Langevine dynamics in [[1]][#reference-1].

## Denoising Score Matching
In order to obtain samples from $`p_{data}(x)`$, we can train a score network such that such that $`\mathbf{s}_{\theta}(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})`$ and then approximately obtain samples with Langevin dynamics using $`\mathbf{s}_{\theta}(\mathbf{x})`$. This is the key idea of
our framework of score-based generative modeling. [[2]][#reference-2]

we can find the best model of the score function by optimizing the following regression problem [[3]][#reference-3]
```math
\ell(\theta)=\frac{1}{2} \int \Big\| \mathbf{s}_{\theta}(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x}) \Big\|^2 \, p_{\text{data}}(\mathbf{x}) \, d\mathbf{x}
```

The problem is that it is not computed since the true $`p_{\text{data}}(\mathbf{x})`$ is unknown. Denoising score matching is a solution by adding a small pre-specified noise $`\mathbf{q}_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})`$ to data and then employs score matching to estimate the score of the perturbed data distribution $`q_{\sigma}(\tilde{\mathbf{x}}) \triangleq \int q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x})\, p_{\text{data}}(\mathbf{x}) \, d\mathbf{x}`$. The objective was proved equivalent to the following: [[4]][#reference-4]
```math
\ell_\mathbf{q_{\sigma}}(\theta)=\frac{1}{2}\,\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}(\mathbf{x}),\,\tilde{\mathbf{x}}\sim q_{\sigma}(\cdot\,|\,\mathbf{x})}
\Big[ \big\| \mathbf{s}_{\theta}(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}}\log q_{\sigma}(\tilde{\mathbf{x}}\,|\,\mathbf{x}) \big\|_2^{2} \Big]
```
As a result, the optimal score network $`\mathbf{s}_\theta^{*}(\mathbf{x})`$ satisfies $`\mathbf{s}_{\theta^{*}}(\mathbf{x}) = \nabla_{\mathbf{x}} \log q_{\sigma}(\mathbf{x})`$ almost surely. As long as the noise is small enough, we can consider that $`q_{\sigma}(\mathbf{x}) \approx p_{\text{data}}(\mathbf{x})`$.

## Annealed Langevin Dynamics
To further improve the sample quality, we can use annealed Langevin dynamics. [[2]][#reference-2] propose to improve score-based generative modeling by 
1) Perturbing the data using various levels of noise; 
2) Simultaneously estimating scores corresponding to all noise levels by training a single conditional score network.

The idea is to gradually decrease the noise level $`\sigma`$ during sampling. At each noise level, we run several steps of Langevin dynamics using the corresponding score network $`\mathbf{s}_{\theta}(\mathbf{x}, \sigma)`$ trained at that noise level. This helps smoothly transfer the benefits of large noise levels to low
noise levels where the perturbed data are almost indistinguishable from the original ones.

## Training
We choose the noise distribution to be $`q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}} \mid \mathbf{x}, \sigma^2 \mathbf{I})`$. Therefor $`\nabla_{\tilde{\mathbf{x}}} \log q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x}) = -(\tilde{\mathbf{x}} - \mathbf{x}) / \sigma^2`$.  For a given σ, the denoising score matching objective is:
```math
\ell(\theta; \sigma) \triangleq \frac{1}{2} \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \mathbb{E}_{\tilde{\mathbf{x}} \sim \mathcal{N}(\mathbf{x}, \sigma^2 \mathbf{I})} \left[ \left\| \mathbf{s}_{\theta}(\tilde{\mathbf{x}}, \sigma) + \frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2} \right\|_2^2 \right]
```
Let $`\{\sigma_i\}_{i=1}^L`$ be a positive geometric sequence that satisfies $`\frac{\sigma_1}{\sigma_2} = \cdots = \frac{\sigma_{L-1}}{\sigma_L} > 1`$ [[2]][#reference-2]. Considering all the $`\sigma \in \{\sigma_i\}_{i=1}^L`$, the unified objective is:
```math
\mathcal{L}(\theta; \{\sigma_i\}_{i=1}^L) \triangleq \frac{1}{L} \sum_{i=1}^L \lambda(\sigma_i) \, \ell(\theta; \sigma_i)
```
where $`\lambda(\sigma_i)>0`$ is a coefficient function depending on $`\sigma_i`$. Assuming $`\mathbf{s}_{\theta}(\mathbf{x}, \sigma)`$ has enough capacity, $`\mathbf{s}_{\theta^{*}}(\mathbf{x}, \sigma)`$ minimizes the above objective if and only if $`\mathbf{s}_{\theta^{*}}(\mathbf{x}, \sigma_i)=\nabla_{\mathbf{x}} \log q_{\sigma_i}(\mathbf{x})`$ a.s. for all $`i \in \{1, 2, \dots, L\}`$.

## Inference
After $`\mathbf{s}_{\theta}(\mathbf{x}, \sigma)`$ is trained, we can use annealed Langevin dynamics to produce samples as shown in the following algorithm [[2]][#reference-2]:

### Algorithm: Annealed Langevin Dynamics

**Require:** $`\{\sigma_i\}_{i=1}^L, \ \epsilon, \ T`$
1. Initialize $`\tilde{\mathbf{x}}_0`$
2. **for** $` i \leftarrow 1 \ \text{to} \ L `$ **do**
3. $`\quad \alpha_i \leftarrow \epsilon \cdot \sigma_i^2 / \sigma_L^2 `$  ⟶ $`\alpha_i`$ is the step size.
4. $`\quad`$ **for** $` t \leftarrow 1 \ \text{to} \ T `$ **do**
5. $`\qquad`$ Draw $` \mathbf{z}_t \sim \mathcal{N}(0, \mathbf{I}) `$
6. $`\qquad \tilde{\mathbf{x}}_t \leftarrow \tilde{\mathbf{x}}_{t-1} + \frac{\alpha_i}{2} s_{\theta}(\tilde{\mathbf{x}}_{t-1}, \sigma_i) + \sqrt{\alpha_i} \mathbf{z}_t `$
7. $`\quad`$ **end for**
8. $`\quad \tilde{\mathbf{x}}_0 \leftarrow \tilde{\mathbf{x}}_T`$
9. **end for**
10. **return** $`\tilde{\mathbf{x}}_T`$




## Reference
[1] [A Simplified Overview of Langevin Dynamics](https://friedmanroy.github.io/blog/2022/Langevin/)

[#reference-1]: https://friedmanroy.github.io/blog/2022/Langevin/

[2] [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)

[#reference-2]: https://arxiv.org/abs/1907.05600

[3] [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)

[#reference-3]: https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf

[4] [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)

[#reference-4]: https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf