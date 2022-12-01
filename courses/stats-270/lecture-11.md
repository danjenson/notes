---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 11: Metropolis-Hastings Algorithm (2022-11-01)"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

## Metropolis Algorithm

- Consider a disputed paper of length $W=2000$. Fixed word $n$, let $y$ be the
  count of this word in the paper. $\lambda_n$ is the log ratio.

$$
\begin{aligned}
e^{\lambda_n}
&= \frac{P(y\mid H)}{P(y\mid M)} \\
&= \frac{\int P(y\mid \mu_H)P(\mu_H,\mu_M\mid\vec{x})\dd \mu_H\dd\mu_M}{\int P(y\mid \mu_M)P(\mu_H,\mu_M\mid\vec{x})\dd \mu_H\dd\mu_M} \\
\vec{x}
&= \left\{x_{ij}, i=M\text{ or }H,j=1,\ldots,J_i\right\} \\
J_i
&= \text{number of known papers by author }i \\
\end{aligned}
$$

- Summing over $N$ words would give $e^{\sum_{i=1}^n\lambda n}$
- $P(y\mid H)=\int h(\mu_H)\pi(\mu_H,\mu_M)\dd
  \mu_H\dd\mu_M=\mathbb{E}_{\pi}\left[h(\mu_M)\right]$ where
  $\pi(\mu_H,\mu_M)=P(\mu_H,\mu_M\mid \vec{x})$ and
  $h(\mu)=\frac{(2\mu)^y}{y!}e^{-2\mu}$ (2 because the rate is per 1000 and
  there are 2000 words).
- Recall that $\sigma=\mu_H+\mu_M$ and $\lambda=\frac{\mu_H}{\mu_H+\mu_M}$,
  assuming $\sigma\sim \operatorname{Uniform}\left(\cdot,\cdot\right)$ and
  $\tau\sim \operatorname{Beta}\left(\gamma,\gamma\right)$ and
  $\gamma=\beta_1+\beta_2\sigma$.
- The joint density (which is the posterior) is then

$$
\begin{aligned}
\pi(\mu_H,\mu_M)
&= \underbrace{c\left[\left(\frac{\mu_H}{\mu_H+\mu_M}\right)\left(\frac{\mu_M}{\mu_H+\mu_M}\right)\right]^{\beta_1+\beta_2(\mu_H+\mu_M)-1}}_{\text{prior}}\cdot\underbrace{\left[\frac{\mu_H}{(\mu_H+\mu_M)^2}\right]}_{\text{Jacobian}}\cdot
\underbrace{
\left[\mu_H^{\left(\sum_{i=1}^{J_i} x_{H_j}e^{- \mu_H\sum_{i=1}^{J_i}w_{H_j}}\right)}\right]
\cdot\left[\mu_H^{\left(\sum_{i=1}^{J_i} x_{M_j}e^{- \mu_M\sum_{i=1}^{J_i}w_{M_j}}\right)}\right]}_{\text{likelihood}}
\end{aligned}
$$

- This is still relatively simple and has clear sufficient statistics.
- Now consider the Negative Binomial, which will have more variance in
  proportion to the value of $\delta$:

$$
\begin{aligned}
P_{nb}(y\mid w\mu, w\delta)
&=\frac{\Gamma(x+k)}{x!\Gamma(k)}(w\delta)^x(1+w\delta)^{-(x+k)} \\
k &= \frac{\mu}{\delta} \\
\mathbb{E}\left[Y\right]
&= w\mu  \\
\operatorname{var}\left[Y\right]
&= w\mu(1+w\delta)
\end{aligned}
$$

- Now there is no simplification due to sufficient statistics (there are no
  sufficient statistics other than the entire data), and the integral is
  4-dimensional.
- $\mathbb{E}_{\pi}\left[h(\mu_H,\delta_H; \mu_M,\delta_M)\right]$
- Now you need MCMC to evaluate this integral. (H&M didn't have MCMC, and ended
  up solving it by asymptotic expansion).

## Formalization

- Suppose we have prior $\pi_0(\theta)$ where $\theta\in \mathbb{R}^k$ and $k$
  can be large.
- The posterior is $\pi(\theta)\propto\pi_0(\theta)f_\theta(y)$ and can be
  evaluated at any point $\theta$.
- To study it, you can
  - Plot it
  - Numerically integrate to get $\mathbb{E}_{\pi}\left[g(\theta)\right]$ (if
    $g(\theta)$ is the indicator function, you get the posterior).
  - These won't work if $\theta$ is high-dimensional.
- Instead, we use sampling to extract information from the posterior.
  - Generate a sequence of values,
    $\theta^{(1)},\theta^{(2)},\ldots,\theta^{(n)}$, each having $\pi(\theta)$ as
    its density.
  - Estimate $$\mathbb{E}_{\pi}\left[g(\theta)\right]=\int
  \pi(\theta)g(\theta)\dd\theta$$ by $$\frac{1}{n}\sum_{i=1}^n g(\theta^{(i)})$$
  - By the Law of Large numbers the sample average will converge to the
    population average with iid samples.
- iid sampling is difficult in general, but if we allow the sample to have
  Markov dependency, then there are good algorithms to solve this.

## MCMC

- **Goal**: generate $x_1, x_2,\ldots x_n$ by evolving a Markov Chain $x_t\sim P(\cdot\mid x_{t-1})$ so that:
  1. $$x_t\sim \pi(\cdot)$$ when $t$ is large.
  2. For any "nice" function $h$, we have $$\underbrace{\frac{1}{n}\sum_{i=1}^n h(x_t)}_{\text{time average}}\to
     \mathbb{E}_{\pi}\left[h(x)\right]=\underbrace{\int h(x)\pi(x)\dd
     x}_{\text{space average}}$$

## Review of Markov Chains

- Book: Durrett, Chapter 5
- Let $x_0,x_1,\ldots,x_n$ be a Markov chain with state space $\mathcal{X}$
  (assume $\mathcal{X}$ is countable) and transition kernel
  $k(x,y)=P(X_{t+1}=y\mid X_t=x)$
- Definitions:
  - $y$ is reachable from another state $x$ if $P_x(\text{waiting time to hit }y<\infty) > 0$.
  - $x$ is recurrent if $P_x(\text{waiting time to return to }x<\infty)=1$.
  - $P_x(T_x<\infty)=1$ where $T_x$ is the waiting time to return to $x$.
- A recurrent state $x$ is a positive recurrent state if
  $\mathbb{E}_{x}\left[T_x\right]<\infty$. (It is just recurrent if the
  expectation is infinite).
- A density $\pi(\cdot)$ on $\mathcal{X}$ is "invariant" if $$\sum_{x}
  \underbrace{\pi(x)}_{\text{density for }x_t} k(x,y)=\underbrace{\pi(y)}_{\text{density for }x_{t+1}}\;\forall y\in \mathcal{X}$$
  - $\pi$ is the invariant density (not all densities will have this behavior).
- **Basic theorem**: If $$\{X_n\}_{n=1}$$ is irreducible, i.e. every state is
  reachable from every other state, then the following are equivalent
  conditions:
  1. Some $x$ is positive recurrent.
  2. All $x$ are positive recurrent.
  3. There is a unique invariant distribution $\pi$, i.e. $\pi(y)\propto
     \frac{1}{\mathbb{E}_{y}\left[T_y\right]}$.
- Also, for any $h(\cdot)$, $\frac{1}{n}\sum_{i=1}^n h(X_t)\to
  \mathbb{E}_{\pi}\left[h(X)\right]$ (time average converges to space average),
  regardless of where you start/the initial value.
- How do we construct a Markov chain that is guaranteed to have $\pi(\cdot)$ as
  its invariant density?
  - Answer: satisfy detailed balance.
- **Detailed balance**:

  - let $\pi(\cdot)$ be a density and $k(x,y)$ be a transition kernel, then
    $(\pi, k)$ satisfy detailed balance if

  $$
  \begin{aligned}
  \pi(x)k(x,y)
  &=\pi(y)k(y,x)\;\forall x,y
  \end{aligned}
  $$

- If detailed balance holds, then we get $\pi(\cdot)$ is invariant under the
  transition kernel $k$.
- **Proof**:

$$
\begin{aligned}
\int\pi(x)k(x,y)\dd x
&= \int \pi(y)k(y,x)\dd x \\
&= \pi(y)\int k(y,x)\dd x \\
&= \pi(y)\cdot 1 \\
\end{aligned}
$$

## Metropolis-Hastings Algorithm

- Let $q(x,y)$ be a "proposal" transition kernel.
- For $t=1,2,\ldots$
  1. Draw $y$ from the proposal $q(x_{t-1}, y)$ (let $x=x_{t-1}$)
  2. Compute the MH ratio, $r=\min \left[1, \frac{\pi(y)q(x,y)}{\pi(x)q(y,x)}\right]$.
  3. Draw $$U\sim \operatorname{Uniform}\left(0,1\right)$$, set $$x_t=\begin{cases}
    y & \text{if }u < r \\
    x_{t-1} &\text{otherwise} \\
  \end{cases}\quad$$
  - In other words, accept the proposal value with probability $r$. And we
    claim that this Markov chain satisfies detailed balance.
- Note: only need to evaluate $\pi(\cdot)$ up to a proportionality constant
  because the normalization constant is cancelled out in the MH ratio.

## Example

- Proposal distribution is a random walk.
- $\pi(x)=e^{-\lambda}\frac{\lambda^x}{x!}$ for $x=0,1,2,\ldots$
- Let $$q(x,y)=\begin{cases}
  y=\begin{cases}
  x+1 & \text{ with p}=1/2 \\
  x-1 & \text{ with p}=1/2
  \end{cases} & \text{if }x > 0 \\
  y=x+1 &\text{with p}=1 \text{ if }x=0 \\
  \end{cases}
  $$
- Ratio of target: $\frac{\pi(y)}{\pi(x)}=\frac{\lambda^y x!}{\lambda^x y!}$
- Ratio of proposal: if $x,y\ge 1$, then $\frac{q(y,x)}{q(x,y)}=1$.
  - If $x=0,y=1$, then ratio is $1/2$.
  - If $x=1,y=0$, then ratio is $2$.
