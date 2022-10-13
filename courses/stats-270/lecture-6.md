---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 6: Likelihood and Conditionality Principles"
toc: true
---

# Likelihood Principle

- R.A. Fisher:

  - Sufficiency
  - Likelihood
  - Analysis of variance (ANOVA)
  - Maximum Likelihood Estimate (MLE)
  - F-statistics
  - Efficiency
  - p-value
  - Design of experiments
  - Fiducial inference (uses only likelihood for inference, no longer really
    used)

- **Likelihood principle**: Let $Y\sim f_\theta(\cdot)$, $\theta\in\Omega$,
  $Y_1$ and $Y_2$ are 2 possible observations giving the same likelihood
  function, then inference on $\theta$ based on one of $Y_1$ should be identical
  to to the inference based on $Y_2$. Likelihoods have the same "shape," i.e.
  the same up to a constant.
- **Theorem**: The likelihood principle is equivalent to the sufficiency
  principle.
- **Lemma**: The likelihood function induces the minimal sufficient partition.
- Proof:

  - (a) The partition induced by the likelihood is a sufficient partition.
    - What is $$\mathcal{Y}_{L(y_0)}=\{y\in \mathcal{Y}: y\text{ gives the same
    likelihood as }y_0\}$$
    - Answer: $Y_1,Y_2$ given the same likelihood iff
      $f_\theta(Y_1)/f_\theta(Y_2)=h(Y_1,Y_2)$ (no longer depends on $\theta$ then
      $$
      \mathcal{Y}_{y_0}=\{y\in \mathcal{Y}:
      f_\theta(y)/f_\theta(y_0)=h(y,y_0)\}
      $$
    - Claim: $\mathcal{Y}=\cup_y \mathcal{Y}_{L(y)}$ is a sufficient partition.
    - Proof: For any $y\in \mathcal{Y}_{L(y)}$,
      $$
      \begin{aligned}
      \Pr_\theta(y\mid
      \mathcal{Y}_{L(y_0)})
      &=\frac{f_\theta(y)}{\int_{\mathcal{Y}_{L(y)}} f_\theta(y)\dd y}
      \\ &=\frac{f_\theta(y_0)h(y,y_0)}{\int_{\mathcal{Y}_{L(y_0)}}f_\theta(y_0)h(y,y_0)\dd y}
      \\ &=\frac{h(y,y_0)}{\int_{\mathcal{Y}_{L(y_0)}}h(y,y_0)\dd y}
      \end{aligned}
      $$
    - TODO last line above
  - (b) Let $S$ be a sufficient statistic, $\mathcal{Y}_s=\\{y: S(y)=s\\}$, and
    you want to show that $\mathcal{Y}=\cup_s \mathcal{Y}_s$ is not coarser than
    likelihood partition.

    - Choose $Y_1,Y_2$ both in the same slice $\mathcal{Y}_s$.
    - You want to show that $Y_1,Y_2$ belong to the same slice in the likelihood
      partition. Does $\frac{f_\theta(y_1)}{f_\theta(y_2)}$ depend on $\theta$?

    $$
    \begin{aligned}
    \frac{f_\theta(y_1)}{f_\theta(y_2)}
    &= \frac{\Pr_{s,\theta}(s)\Pr(Y_1\mid S=s)}{\Pr_{S,\theta}(s)\Pr(Y_2\mid S=s)}\quad\text{no $\theta$ because
    sufficient statistic}
    \\ &= \frac{\Pr(Y_1\mid S=s)}{\Pr(Y_2\mid S=s)}
    \end{aligned}
    $$

    - Therefore, $Y_2, Y_2$ induces the same likelihood.
    - TODO add picture

# Conditionality Principle

- **Definition**: A statistic $C=C(Y)$ is an ancillary statistic if it has the
  same marginal distribution under all all densities $f\in \mathcal{F}$.
- Example 1:
  - $N\sim 1+\operatorname{Poisson}\left(5\right)$
  - Given $N=n$, $Y_1,Y_2,\ldots,Y_n\sim \operatorname{Bernoulli}\left(\theta\right)$
  - Then $N$ is ancillary, i.e. the marginal distribution of $N$ is independent of
    $\theta$.
  - $\mathcal{L}(\theta; n,y_1,\ldots,y_n)=c\theta^{\sum_{i=1}^n
  y_i}(1-\theta)^{n-\sum_{i=1}^n y_i}$
  - Then the sufficient statistic is $S=(n, \sum\_{i=1}^n Y_i)$ and it is
    minimal.
- **Conditionality Principle**: If $C$ is ancillary, then inference should be
  based on the conditional distribution of $Y$ given $C=c$. (Can be
  conceptualized as reverse of sufficiency).
- Intuition: 2-stage experiment
  1. Draw $C=c$ from the marginal density $f_C(\cdot)$, which does not depend on
     $\theta$.
     - Does not contain any information on $\theta$.
  2. Draw $Y=y$ from the conditional distribution of $Y\mid C=c$, $f_{Y\mid C=c;
     \theta}(\cdot)$
- TODO drawing
- Bayesian inference obviously obeys the conditionality principle.
- However, this is not true for most other approaches to inference.
- Example: Construct confidence interval based on MLE
  1. Find MLE $\hat{\theta}(Y)$ by maximizing $\mathcal{L})(\theta; Y)$
  2. Consider $\mathcal{D}_\theta(\hat{\theta}(Y)-\theta)$ in order to find the
     $(1-\alpha)$ C.I., $\hat{\theta}\pm\operatorname{SE}$
  - Now, suppose $C(Y)$ is ancillary, and $S=(T(Y), C(Y))$ is the minimal
    sufficient statistic ($T(Y)$ is the sum in previous example).
  - Then $\Pr_\theta(Y)=\Pr_\theta(T\mid C)\Pr(C)$, so MLE $\tilde{\theta}$ is
    the same whether you condition on $C$ or not, but
    $\mathcal{D}_\theta(\hat{\theta}-\theta)\ne
    \mathcal{D}\_\theta(\hat{\theta}(Y)-\theta\mid C=c)$
  - TODO note?
- In Example 1:
  - $N=1+\operatorname{Poisson}\left(5\right)$
  - $\mathcal{L}(\theta\mid Y)\propto \theta^{\sum_{i=1}^nY_i}(1-\theta)^{N-\sum_{i=1}^nY_i}$
  - $\hat{\theta}^{\operatorname{MLE}}=\frac{1}{N}\sum_{i=1}^N Y_i$, what is the
    distribution of $\hat{\theta}-\theta$?
  - Without conditioning:
    - $\mathbb{E}\left[\hat{\theta}\right]=\mathbb{E}\left[\mathbb{E}\left[\hat{\theta}\mid N\right]\right]=\mathbb{E}\left[\theta\right]=\theta$
    - $\operatorname{var}\left[\hat{\theta}\right]=\mathbb{E}\left[\operatorname{var}\left[\hat{\theta}\mid N\right]\right]+\operatorname{var}\left[\mathbb{E}\left[\hat{\theta}\mid N\right]\right]=\mathbb{E}\left[\frac{\theta(1-\theta)}{N}\right]+0=\theta(1-\theta)\mathbb{E}\left[\frac{1}{N}\right]$
  - With conditioning, given $N=n$
    - $\mathbb{E}\left[\hat{\theta}\mid N=n\right]=\theta$
    - $\operatorname{var}\left[\frac{\theta(1-\theta)}{n}\right]=\theta(1-\theta)\frac{1}{n}$
  - These could be very different depending on how much $N=n$ differs from the
    expectation.
  - For $N\sim 1+\operatorname{Poisson}\left(5\right)$, each event $$A=\{N\le
  2\}$$, or $$B=\{N\ge 8\}$$ has non-negligible probability, but the degree of
    uncertainty can be very different.
- In his critique of unconditional C.I. theory, Fisher pointed out that there is
  the problem / phenomenon of "relevant subset."
  - Suppose $A(Y)$ is a $(1-\alpha)$ confidence set, i.e. $\Pr_\theta(\theta\in
  A(Y))\equiv 1-\alpha$ and $B\in \mathcal{Y}$ is a set satisfying either:
    - (a) $\sup_{\theta\in\Omega}\Pr_\theta(\theta\in A(Y)\mid Y\in B)<1-\alpha$
    - (b) $\inf_{\theta\in\Omega}\Pr_\theta(\theta\in A(Y)\mid Y\in B)>1-\alpha$
  - Here $B$ is called a relevant subset.
  - Do not use $1-\alpha$ if you know $Y\in B$.
  - Unfortunately, relevant subset can exist even in the nicest parametric model
    such as iid observations $\sim
  \operatorname{Normal}\left(\mu,\sigma^2\right)$.
- Another problem is that ancillary statistics are not unique.
- Example 2 (Basu):

  $$
  Y_1,\ldots,Y_n\sim \operatorname{Categorical}\left(\begin{bmatrix}\frac{1}{6}(1-\theta) \\ \frac{1}{6}(1+\theta) \\ \frac{1}{6}(2-\theta) \\ \frac{1}{6}(2+\theta)\end{bmatrix}\right)
  $$

- Sufficient statistic is $$N=\begin{bmatrix}N_1\\N_2\\N_3\\N_4\end{bmatrix}$$,
  the counts.
- Ancillary statistics:
  - $$
    C=\begin{bmatrix}N_1+N_2\\
    N_3+N_4\end{bmatrix}=\begin{bmatrix}\frac{1}{3}\\\frac{2}{3}\end{bmatrix}\sim
    \operatorname{Multinomial}\left(n,\begin{bmatrix}\frac{1}{3}\\\frac{2}{3}\end{bmatrix}\right)
    $$
  - $$
    D=\begin{bmatrix}N_1+N_4\\ N_2+N_3\end{bmatrix}\sim
    \operatorname{Multinomial}\left(n,\begin{bmatrix}\frac{1}{2}\\\frac{1}{2}\end{bmatrix}\right)
    $$
  - Hard to decide between these two ancillary statistics.
  - Everything is satisfied when you use a prior in a Bayesian framework, i.e.
    sufficiently, likelihood, conditionality, etc.
