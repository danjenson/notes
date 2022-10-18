---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 7: Non-informative Priors"
toc: true
---

# Non-informative Priors

- Example: iid variables from $\operatorname{Bernoulli}\left(\theta\right)$
  where $\theta\in[0,1]$
- Bayes & Laplace used uniform priors to represent state of "no information."
  - Put the same probability for $[\theta_0\pm\varepsilon]$ and
    $[\theta_1\pm\varepsilon]$
  - This was popular for about 100 years before being criticized.
  - This was criticized by:
    1. The subjective Bayesian (Ramsey, de Finetti, Savage).
    2. Frequentists (Neyman, Fisher).
- Objections to the uniform prior:
  - Suppose $\theta\sim \operatorname{Uniform}\left(0,1\right)$ to represent "no
    information on $\theta$."
  - Now, $\phi=-\log(\theta)\implies \theta=e^{-\theta}$
  - If we use $\phi$ as the parameter, we should also have "no information."
  - A uniform prior on $\phi$ is not equivalent to a uniform prior on $\theta$.
    - If $\theta\sim \operatorname{Uniform}\left(0,1\right)$, then $\phi\sim
    \operatorname{Exponential}\left(1\right)$
    - On which scale should we put a uniform prior?
  - Suppose there a $\phi(\theta)$ for which the uniform prior is "correct."
  - TODO: picture
  - So equal probability on $$\{\phi\in\phi_0\pm\varepsilon\}$$ and
    $$\{\phi\in\phi\pm\varepsilon\}$$ implies that $$\{\theta\in\theta_0\pm
    \sigma(\theta_0)\varepsilon\}$$ has same probability as
    $$\{\theta\in\theta_1\pm\sigma(\theta_1)\varepsilon\}$$
    - $$\pi(\theta_0)\sigma(\theta_0)\varepsilon=\pi(\theta_1)\sigma(\theta_1)\varepsilon$$,
      then putting $$\pi(\theta)\propto\frac{1}{\sigma(\theta)}$$
- What is a good "yardstick" for measuring distance in $\theta$?
  - Consider $\operatorname{Bernoulli}\left(\theta\right)$, if we have $n$
    observations, then $p(\theta\mid
  x_1,\ldots,x_n)\propto\pi(\theta)\theta^{n\bar{x}}(1-\theta)^{n(1-\bar{x})}$
    - If $n$ is large, then the posterior is
      $\theta^{n\theta_0}(1-\theta)^{n(1-\theta_0)}$, this is a
      $\operatorname{Beta}\left(n\theta_0,n(1-\theta_0)\right)$, then the
      posterior mean is $\theta_0$ and posterior variance is
      $\sqrt{\theta_0(1\theta_0)/n}$, then
      $\operatorname{sd}\propto\sqrt{\theta_0(1-\theta_0)}$
    - This suggests that $\sigma(\theta_0)=\sqrt{\theta_0(1-\theta_0)}$.
    - The non-informative prior is then
      $$\pi^*(\theta)\proprto\frac{1}{\sqrt{\theta(1-\theta)}}$$

# Jeffrey's Prior

- Let $x_1,\ldots,x_n$ be iid for $f_\theta(\cdot)$.
- Define the score function as $\pdv{\theta}\log f_\theta(x)=S(\theta,x)$
- Fisher information $i(\theta)=\operatorname{Var}(S(\theta,x))=E(S^2)$
- Von Mises Theorem: If the sample size is large, then $\theta\mid
x_1,\ldots,x_n\sim
\operatorname{Normal}\left(\theta_0,1/(n(i(\theta)))\right)$, which is true
  regardless of the prior.
- So, the correct measure is $\sigma(\theta)\propto i(\theta)^{-1/2}$.
- $$\pi^*(\theta)\propto i(\theta)^{1/2}$$, which is Jeffrey's prior.
- Exercise: Check this on the $\operatorname{Bernoulli}\left(\theta\right)$
- Geometric view of Jeffrey's Prior:
  - Let $$\mathcal{F}=\{f_\theta(\cdot),\theta\in[0,1]\}$$ where $\mathcal{X}$
    is arbitrary.
  - Define $\sqrt{x}\mid\theta=\sqrt{f_\theta(x)}$
  - $\langle v_1,v_2\rangle=\int_\mathcal{X}v_1(x)v_2(x)\dd x$
  - $$\mathcal{G}=\{v_\theta:\theta\in\Omega\}$$ is a curve in $L_2(\mathcal{X})$
  - It is natural to use the arc-length as the distance measure.
  - TODO picture
  - This suggests that $\pi(\theta)\cdot\Delta\propto
  ||v_{\theta+\Delta}-v_\theta||$
  - $\pi(\theta)\propto \lim_{\Delta\to 0}||v_{\theta+\Delta}-v_\theta||$
  - Bernoulli example:
    - $$\mathcal{X}=\{x_1,x_2\}=\{0,1\}$$
    - $$f_\theta(\cdot)=\begin{bmatrix}\theta \\ 1-\theta\end{bmatrix}$$
    - $$v_\theta=\frac{\sqrt{\theta}}{\sqrt{1-\theta}}$$,
      $L_2(\mathcal{x})=\mathbb{R}^2$
    - Then, $$\pi^*(\theta)=\lim_{\Delta\to
    0}\frac{||v_{\theta+\Delta}-v_\theta||}{\Delta}=\frac{1}{2}\cdot\frac{1}{\sqrt{\theta(1-\theta)}$$,
      which is exactly Jeffrey's prior.
      - If $\mathcal{X}$ has $k$ points, then $v_\theta\in \mathbb{R}^k$
      - If $\mathcal{X}\in[0,1]$, then $v_\theta(\cdot)=\sqrt{f_\theta(\cdot)}$
        is a $L_2$ function in $[0,1]$.
    - Heddinger(?) distance between two densities.
- Multi-dimensional case:
  - $\theta\in\Omega$ is a bounded region in $\mathbb{R}^2$
  - TODO picture
  - Transform region from $\Omega$ into $L_2(\mathcal{X})$
    - $v_{\theta_1,\theta_2}\to v_{\theta_1+\delta_1,\delta_2}$
    - You uniform in $L_2$ space and then translate it back.
    - So, $\pi(\theta_1,\theta_2)=\operatorname{Area}[v^{(1)},v^{(2)}]$
    - $\operatorname{Vol}=\sqrt{\det(G)}$ where $G$ is the Graham matrix.
    - $$\pi(\theta_1,\theta_2)=\operatorname{Area}[v^{(1)},v^{(2)}]=\left|\det \begin{bmatrix}\langle<v^{(1)},v^{(1)}\rangle & \langle<v^{(1)},v^{(2)}\rangle \\ \langle<v^{(2)},v^{(1)}\rangle & \langle<v^{(2)},v^{(2)}\rangle  \end{bmatrix}\right|$$
    - TODO picture
    - $v^{(1)}(x)=L_2$-limit of $\frac{1}{\Delta_1}(\sqrt{f_{\theta_1+\delta_1,\theta_2}(x)}-\sqrt{f_{\theta_1,\theta_2}(x)})$ converges to $\pdv{\theta_1}\sqrt{f_{\theta_1,\theta_2}(x)}$
    - Worked out TODO finish
      $$
      \langle v^{(1)},v^{(2)}\rangle
      &=\int_\mathcal{X} TODO finish //
      &=\frac{1}{4} \mathbb{E}\left[\pdv{\theta_1}\log f\cdot \pdv{\theta_2}\log f\right]
      $$
    - In general, for $\theta$ that is $k$-dimensional, we define the score
      function TODO danj notation
    - An important property of the score function is $\mathbb{E}_\theta\left[\ell_\theta(x)\right]=0$
    - Fisher-information is now a $k\times k$ matrix.
      - TODO image
    - Then Jeffrey's prior is defined as
      $\pi(\theta)\propto|\det(i(\theta))|^{1/2}$
