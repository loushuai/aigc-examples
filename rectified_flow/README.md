# Rectified Flow

Given two distributions, one is the unknown image distribution $`\pi_0`$ and the other one $`\pi_1`$ is a simple distribution like Gaussian, the rectified flow aim to find a transport map from $`\pi_1`$ to $`\pi_0`$ with high computational efficiency.

## Optimal Transport
Optimal transport(OT) tries to find special couplings that are optimal in therms of minimizing a transport cost. In particular, Monge's OT problem is: [[1]][#reference-1]
```math
\min_{T} \mathbb{E}\big[c(T(Z_0) - Z_0)\big] \quad \text{s.t.} \quad p(Z_0) \sim \pi_0,\; p(T(Z_0)) \sim \pi_1
```
where $`c : \mathbb{R}^d \to \mathbb{R}`$ is a cost function, e.g., $`c(x) = \frac{1}{2} \|x\|^2`$, and the $`\mathbb{E}\big[ c\big(T(Z_0) - Z_0\big) \big]`$ measures the expected effort of transporting $`Z_0`$ and $`Z_1=T(Z_0)`$.
However, solving OT is very challenging and it remains open to develop efficient algorithms for high dimensional and big data settings. We need to find alternative way that are more directly related to ML tasks and easier to enforce in practice.

## Rectified flow
Rectified flow learns the transport map implicitly by constructing an ordinary differential equation (ODE) driven by a drift force $`v : \mathbb{R}^d \times [0,1]`$: [[1]][#reference-1]
```math
\mathrm{d}Z_t = v(Z_t, t)\,\mathrm{d}t,\quad t \in [0,1], \text{ starting from } Z_0 \sim \pi_0
```
such that $`Z_1 \sim \pi_1`$ when following the ODE starting from $`Z_0 \sim \pi_0`$. The main problem is how to construct the drift based on observations from $`\pi_0`$ and $`\pi_1`$, presumably using deep neural networks or other nonlinear approximators.
One natural approach is to find $`v`$ by minimizing $`D\big(\rho_1^v ; \pi_1\big)`$, where $`\rho_1^v`$ is the distribution of $`Z_1`$ following the ODE with $`v`$ and $`D(\cdot \, ; \, \cdot)`$ is a discrepancy measure, such as KL divergence. However, inferring (i.e., sampling or calculating the likelihood of) $`\rho_1^v`$ requires repeated simulation of the ODE, which is computationally expensive. The trouble here is that we do not know what intermediate trajectories the ODE should travel through before hand and hence need to infer it repeatedly.

Fortunately, this difficulty can be avoided by exploiting the over-parameterized nature of the problem: because we are only concerned with having the correct starting and terminal distributions $`\pi_0`$ and $`\pi_1`$, the intermediate distributions $`\pi_t`$ of $`Z_t`$ can be essentially an arbitrary smooth interpolation between $`\pi_0`$ and $`\pi_1`$. Hence, we can (and should) inject very strong priors on the intermediate trajectories, so that we can avoid the need for repeated inference, and also, as a bonus, incorporate proper beneficial properties. Obviously, the simplest prior is straight trajectories. Straight paths are attractive both theoretically as an essential ingredient for achieving optimal transport, and computationally because ODEs with straight paths can be exactly simulated without time discretization.

Specifically, rectified flow works by finding an ODE to match (the marginal distributions of) the linear interpolation of points from $`\pi_0`$ and $`\pi_1`$. Assume we observe $`X_0 \sim \pi_0`$ and $`X_1 \sim \pi_1`$. Let $`X_t`$ for $`\forall t \in [0,1]`$ be the linear (or geodesic) interpolation of $`X_0`$ and $`X_1`$:
```math
X_t = t X_1 + (1 - t) X_0,\quad t \in [0,1]
```
Observe that $`X_t`$ follows a trivial ODE that already transfers $`\pi_0`$ to $`\pi_1`$:
```math
\mathrm{d}X_t = (X_1 - X_0)\,\mathrm{d}t
```
in which $`X_t`$ moves following the line direction $`(X_1-X_0)`$ with a constant speed.

However, this ODE does not solve the problem: it cann’t be simulated causally, because the update $`X_t`$ depends on the final state $`X_1`$, which is not supposed to be known at time $`t<1`$.

Hence, we want to "causalize" the interpolation process $`X_t`$, by "projecting" it to the space of causally simulatable ODEs of form $`\mathrm{d}Z_t = v(Z_t, t)\,\mathrm{d}t`$. A natural way is the L2 projection on the velocity field, finding $v$ by minimizing the least squares loss with the line directions $`X_1 - X_0`$:
```math
\min_{v} \int_{0}^{1} \mathbb{E}\Big[ \big\| (X_1 - X_0) - v(X_t, t) \big\|^2 \Big] \,\mathrm{d}t
```
Theoretically, the solution can be represented using conditional expectation:
```math
v(z,t) = \mathbb{E}\!\left[\,X_1 - X_0 \,\middle|\, X_t = z\,\right]
```
which is the average of the directions of the lines passing through point $`z`$ at time $`t`$.


## Reference
[1] [Rectified flow](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)

[#reference-1]: https://friedmanroy.github.io/blog/2022/Langevin/