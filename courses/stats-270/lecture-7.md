---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 7: Non-informative Priors (2022-10-18)"
toc: true
---

# Non-informative Priors

- Example: iid variables from $\operatorname{Bernoulli}\left(\theta\right)$
  where $\theta\in[0,1]$
- Bayes & Laplace used uniform priors to represent state of "no information."
  - Put the same probability for $[\theta_0\pm\varepsilon]$ and
    $[\theta_1\pm\varepsilon]$ (as long as intervals are of same length, they
    get the same prior probability -- this is the meaning of uniform prior)
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
    {% marginfigure 'uniform-phi-theta' 'courses/stats-270/figures/lecture-7/uniform-phi-theta.png' 'Uniform probability: $\phi$ vs $\theta$.' %}
  - So equal probability on $$\{\phi\in\phi_0\pm\varepsilon\}$$ and
    $$\{\phi\in\phi_1\pm\varepsilon\}$$ implies that $$\{\theta\in\theta_0\pm
    \sigma(\theta_0)\varepsilon\}$$ has same probability as
    $$\{\theta\in\theta_1\pm\sigma(\theta_1)\varepsilon\}$$
    - $$\pi(\theta_0)\sigma(\theta_0)\varepsilon=\pi(\theta_1)\sigma(\theta_1)\varepsilon$$,
      then $$\pi(\theta)\propto\frac{1}{\sigma(\theta)}$$
    - $\sigma(\theta)$ is like a yardstick for measuring distance, like standard
      deviation.
- What is a good "yardstick" for measuring distance in $\theta$ scale?
  - Consider $\operatorname{Bernoulli}\left(\theta\right)$, if we have $n$
    observations, then $p(\theta\mid
  x_1,\ldots,x_n)\propto\pi(\theta)\theta^{n\bar{x}}(1-\theta)^{n(1-\bar{x})}$
    - If $n$ is large, then the posterior converges to
      $\pi(\theta)\theta^{n\theta_0}(1-\theta)^{n(1-\theta_0)}$ (since
      $\bar{x}\to\theta_0$ as $n\to\infty$) where $\pi(\theta)$ becomes
      irrelevant; this is a
      $\operatorname{Beta}\left(n\theta_0,n(1-\theta_0)\right)$, then the
      posterior mean is $\theta_0$ and posterior variance is
      $\sqrt{\theta_0(1-\theta_0)/n}$, then
      $\operatorname{sd}\propto\sqrt{\theta_0(1-\theta_0)}$
    - This suggests that $\sigma(\theta_0)=\sqrt{\theta_0(1-\theta_0)}$.
    - The non-informative prior is then
      $$\pi^*(\theta)\propto\frac{1}{\sqrt{\theta(1-\theta)}}$$

# Jeffrey's Prior

- Let $x_1,\ldots,x_n$ be iid for $f_\theta(\cdot)$.
- Define the score function as $\pdv{\theta}\log f_\theta(x)=S(\theta,x)$
- Fisher information $i(\theta)=\operatorname{Var}(S(\theta,x))=E(S^2)$
  - This is because the score function has mean 0.
- Bernstein-von Mises Theorem: If the sample size is large, then $\theta\mid
x_1,\ldots,x_n\sim
\operatorname{Normal}\left(\theta_0,(ni(\theta))^{-1}\right)$, which is true
  regardless of the prior.
- So, the correct measure is $\sigma(\theta)\propto i(\theta)^{-1/2}$ (square
  root of variance above).
- $$\pi^*(\theta)\propto i(\theta)^{1/2}$$, which is Jeffrey's prior.
- Exercise: Check this on the $\operatorname{Bernoulli}\left(\theta\right)$
- Geometric view of Jeffrey's Prior:

  - Let $$\mathcal{F}=\{f_\theta(\cdot),\theta\in[0,1]\}$$ where $\mathcal{X}$
    is arbitrary.
  - Define the square root density as $v_\theta(x)=\sqrt{f_\theta(x)}$
    - This square root density is a member of the $L_2(\mathcal{X})$ function
      space.
  - $L_2$ space is the inner product space, so $\langle
  v_1(x),v_2(x)\rangle=\int_\mathcal{X}v_1(x)v_2(x)\dd x$
  - $$\mathcal{G}=\{v_\theta(x):\theta\in\Omega\}$$ is a curve in $L_2(\mathcal{X})$
  - It is natural to use the arc-length as the distance measure.
    {% marginfigure 'sqrt-density-curve' 'courses/stats-270/figures/lecture-7/sqrt-density-curve.png' 'Arc length in sqrt density space.' %}
  - This suggests that $\pi(\theta)\cdot\Delta\propto
  \lVert v_{\theta+\Delta}(x)-v_\theta(x)\rVert$
    - This arc length will change depending on where you are in the curve, so
      take the limit as delta approaches 0:

  $$
  \pi(\theta)\propto \lim_{\Delta\to 0}\frac{\lVert v_{\theta+\Delta}(x)-v_\theta(x)\rVert}{\Delta}
  $$

  - Bernoulli example:

    $$
    \begin{aligned}
    \mathcal{X}&=\{x_1,x_2\}=\{0,1\} \\
    f_\theta(x)&=\begin{bmatrix}\theta \\ 1-\theta\end{bmatrix} \\
    v_\theta(x)&=\begin{bmatrix}v_\theta(x) \\ v_{1-\theta}(x)\end{bmatrix} \\
    \end{aligned}
    $$

    - $L_2(\mathcal{X})=\mathbb{R}^2$
    - Then,

    $$
    \begin{aligned}
    \pi^*(\theta)
    &=\lim_{\Delta\to 0}\frac{\lVert v_{\theta+\Delta}(x)-v_\theta(x)\rVert}{\Delta} \\
    &=\frac{1}{2}\cdot\frac{1}{\theta(1-\theta)}
    \end{aligned}
    $$

    - This is exactly Jeffrey's prior.
      If $\mathcal{X}$ has $k$ points, then $v_\theta(x)\in\mathbb{R}^k$
    - If $\mathcal{X}\in[0,1]$, then $v_\theta(x)=\sqrt{f_\theta(x)}$
      is a $L_2$ function in $[0,1]$.

- Multi-dimensional case:
  {% maincolumn 'courses/stats-270/figures/lecture-7/multi-dimensional-prior.png' 'Prior in $\mathbb{R}^2$.' %}

  - The objective is to assign a uniform prior on the $L_2$ space ($B$) and then map
    that probability back to $\theta$-space ($A$).
  - $\theta\in\Omega$ is a bounded region in $\mathbb{R}^2$
  - $\delta_1 v=\sqrt{f_{\theta_1+\Delta_1,\theta_2}(x)}-\sqrt{f_{\theta_1,\theta_2}(x)}$
  - $\frac{\operatorname{Area}(B)}{\Delta_1\Delta_2}=\operatorname{Area}\left[\frac{\delta_1 v}{\Delta_1},\frac{\delta_2 v}{\Delta_2}\right]$
  - Assume $\exists v^{(1)}$ such that $\frac{\delta_1 v}{\Delta_1}\to
  v^{(1)}\text{ as }\Delta_1\to 0$ in $L_2$, then
    $\operatorname{Area}\left[\frac{\delta_1 v}{\Delta_1},\frac{\delta_2
  v}{\Delta_2}\right]\to\operatorname{Area}\left[v^{(1)},v^{(2)}\right]$
  - Because $\operatorname{Area}$ is a continuous function of square root, inner
    product, and the norm, then the area will converge too.
  - $v_{\theta_1,\theta_2}\to v_{\theta_1+\delta_1,\delta_2}$
  - Recall that the density is the probability of a small area divided by its
    area.
  - So, the prior we want $\pi(\theta_1,\theta_2)=\operatorname{Area}[v^{(1)},v^{(2)}]$
  - $\operatorname{Vol}=\sqrt{\det(G)}$ where $G$ is the Graham matrix.
  - $$\pi(\theta_1,\theta_2)=\operatorname{Area}[v^{(1)},v^{(2)}]=\left|\det \begin{bmatrix}\langle v^{(1)},v^{(1)}\rangle & \langle v^{(1)},v^{(2)}\rangle \\ \langle v^{(2)},v^{(1)}\rangle & \langle v^{(2)},v^{(2)}\rangle  \end{bmatrix}\right|^{1/2}$$
  - $v^{(1)}(x)=L_2$-limit of $\frac{1}{\Delta_1}(\sqrt{f_{\theta_1+\Delta_1,\theta_2}(x)}-\sqrt{f_{\theta_1,\theta_2}(x)})$ converges to $\pdv{\theta_1}\sqrt{f_{\theta_1,\theta_2}(x)}$

    $$
    \begin{aligned}
    \langle v^{(1)},v^{(2)}\rangle
    &=\int_\mathcal{X}
    \left(\pdv{\theta_1}\sqrt{f_\theta(x)}\right)\left(\pdv{\theta_2}\sqrt{f_\theta(x)}\right)\dd x \\
    &=\int_\mathcal{X}
    \left(\frac{1}{2}\cdot\frac{\pdv{\theta_1}f_\theta(x)}{\sqrt{f_\theta(x)}}\right)\left(\frac{1}{2}\cdot\frac{\pdv{\theta_2}f_\theta(x)}{\sqrt{f_\theta(x)}}\right)\dd x \\
    &=\frac{1}{4}\int_\mathcal{X}
    \left(\frac{\pdv{\theta_1}f_\theta(x)}{f_\theta(x)}\right)\left(\frac{\pdv{\theta_2}f_\theta(x)}{f_\theta(x)}\right)\cdot f_\theta(x)\dd x \\
    &=\frac{1}{4} \mathbb{E}\left[\pdv{\theta_1}\log f_\theta(x)\cdot \pdv{\theta_2}\log f_\theta(x)\right]
    \end{aligned}
    $$

- In general, for $\theta$ that is $k$-dimensional, we define the score
  function as:

  $$
  \begin{aligned}
  \dot \ell_\theta&=\begin{bmatrix}
  \pdv{\theta_1}\log f_\theta(x) \\
  \vdots \\
  \pdv{\theta_k}\log f_\theta(x) \\
  \end{bmatrix}
  \end{aligned}
  $$

- An important property of the score function is $$\mathbb{E}_\theta\left[\ell_\theta(x)\right]=0$$
- Fisher-information is now a $k\times k$ matrix, which is the
  variance-covariance matrix of $\dot \ell$:

$$
i(\theta)_{i,j}=\mathbb{E}_\theta\left[\dot\ell_i\dot\ell_j\right]
$$

- Then Jeffrey's prior is defined as
  $\pi(\theta)\propto|\det(i(\theta))|^{1/2}$
