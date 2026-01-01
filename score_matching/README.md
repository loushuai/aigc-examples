# Score Matching

## Langevine Dynamics
The core of score matching is Langevine dynamics:
```math
\tilde{\mathbf{x}}_{t} = \tilde{\mathbf{x}}_{t-1} + \frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p(\tilde{\mathbf{x}}_{t-1}) + \sqrt{\epsilon} \mathbf{z}_{t}
```
where $`\mathbf{z}_t \sim \mathcal{N}(0, I)`$. The gradient of the logarithm of a distribution is defined as the score.

Given a fixed step size $`\epsilon>0`$ and an initial value $`\tilde{\mathbf{x}}\_0\sim\pi(\mathbf{x})`$ with $`\pi`$ being an prior distribution, we can use this process to produce samples from a probability density $`p(x)`$ using only the score function $`\nabla_{\mathbf{x}} \log p(\tilde{\mathbf{x}}_{t-1})`$.\
You can find an intuitve introduction of Langevine dynamics in <a name="reference-1"> [1] </a>.

## Denoising Score Matching
In order to obtain samples from $`p_{data}(x)`$, we can train a score network such that such that $`\mathbf{s}_{\theta}(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})`$ and then approximately obtain samples with Langevin dynamics using $`\mathbf{s}_{\theta}(\mathbf{x})`$. This is the key idea of
our framework of score-based generative modeling.<a name="reference-2"> [2] </a>

we can find the best model of the score function by optimizing the following regression problem <a name="reference-3"> [3] </a>
$$\ell(\theta)=\frac{1}{2} \int \Big\| \mathbf{s}_{\theta}(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x}) \Big\|^2 \, p_{\text{data}}(\mathbf{x}) \, d\mathbf{x}$$

The problem is that it is not computed since the true $`p_{\text{data}}(\mathbf{x})`$ is unknown. Denoising score matching is a solution by adding a small pre-specified noise $`\mathbf{q}_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})`$ to data and then employs score matching to estimate the score of the perturbed data distribution $`q_{\sigma}(\tilde{\mathbf{x}}) \triangleq \int q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x})\, p_{\text{data}}(\mathbf{x}) \, d\mathbf{x}`$. The objective was proved equivalent to the following: <a name="reference-4"> [4] </a>
$$\ell_\mathbf{q_{\sigma}}(\theta)=\frac{1}{2}\,\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}(\mathbf{x}),\,\tilde{\mathbf{x}}\sim q_{\sigma}(\cdot\,|\,\mathbf{x})}
\Big[ \big\| \mathbf{s}_{\theta}(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}}\log q_{\sigma}(\tilde{\mathbf{x}}\,|\,\mathbf{x}) \big\|_2^{2} \Big]$$
As a result, the optimal score network $`\mathbf{s}_\theta^{*}(\mathbf{x})`$ satisfies $`\mathbf{s}_{\theta^{*}}(\mathbf{x}) = \nabla_{\mathbf{x}} \log q_{\sigma}(\mathbf{x})`$ almost surely.


## Reference
[[1]](#reference-1) [A Simplified Overview of Langevin Dynamics](https://friedmanroy.github.io/blog/2022/Langevin/)

[[2]](#reference-2) [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)

[[3]](#reference-3) [Estimation of Non-Normalized Statistical Models
by Score Matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)

[[4]](#reference-4) [A Connection Between Score Matching
and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)