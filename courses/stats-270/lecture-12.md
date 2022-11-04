---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 12: Metropolis-Hastings Proof & Gibbs Sampling & Ising Model"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

## Federalist Paper Example

$$
\begin{aligned}
p(y\mid H)
&=\int h(\mu_H)\pi(\mu_H,\mu_M)\dd\mu_M\dd\mu_H \\
\pi(\mu_H,\mu_M)
&=
c\cdot\frac{1}{\operatorname{Beta}\left(\beta_1+\beta_2\sigma,\beta_1+\beta_2\sigma\right)}\left[\left(\frac{\mu_H\mu_M}{\sigma^2}\right)^{\beta_1+\beta_2\sigma}\frac{\mu_H}{\sigma^2}\right]\cdot \left[\mu_H^{\sum_{i=1}^n x_{H_j}}e^{\sum_{i=1}^n w_{H_j}\mu_H}\right]\cdot \left[\mu_M^{\sum_{i=1}^n x_{M_j}}e^{\sum_{i=1}^n w_{M_j}\mu_M}\right] \\
\end{aligned}
$$

- Let the proposal transition be a uniform draw from a box centered on the
  current $\mu=(\mu_H,\mu_M)$
  {% marginfigure 'box-walk' 'courses/stats-270/figures/lecture-12/box-walk.png' 'Random walk in box.' %}

$$
\begin{aligned}
q(u\to u')
&=\begin{cases}
  \frac{1}{4\delta^2}\mathbb{1}_\text{box}\left[\mu'\right] & \text{if }\mu_H,\mu_M>\delta \\
  \frac{1}{2\delta(\mu_{H}+\delta)}\mathbb{1}_\text{box}\left[\mu'\right] & \text{if }0<\mu_H<\delta,\mu_M>\delta \\
  \frac{1}{2\delta(\mu_{M}+\delta)}\mathbb{1}_\text{box}\left[\mu'\right] & \text{if }0<\mu_M<\delta,\mu_H>\delta \\
\end{cases} \\
r
&=\min \left(1,\frac{\pi(\mu')}{\pi(\mu)}\cdot \frac{q(\mu'\to \mu)}{q(\mu\to \mu')}\right)
\\
\end{aligned}
$$

- Accept $\mu'$ with probability $r$.
- This is a Markov chain and would have the stationary distribution $\pi$.
- **Theorem**: Metropolis-Hastings algorithm gives a Markov Chain satisfying detailed balance
  with respect to $\pi(\cdot)$.
- **Proof**: We want to show that detailed balance holds: $\pi(x)k(x,y)=\pi(y)k(y,x)$.
  - If $x=y$, this is trivially satisfied, so assume $x\ne y$.

$$
\begin{aligned}
k(x,y)
&= q(x,y)\cdot\min \left(1,\frac{\pi(y)q(y,x)}{\pi(x)q(x,y)}\right) \\
\end{aligned}
$$

- (a) If both sides are non-negative, $\pi(x)q(x,y)>0$ and $\pi(y)q(y,x)>0$ then

$$
\begin{aligned}
\pi(x)k(x,y)
&=\pi(x)q(x,y)\min \left(1,\frac{\pi(y)q(y,x)}{\pi(x)q(x,y)}\right) \\
&=\min \left(\pi(x)q(x,y),\pi(y)q(x,y)\right) \\
&=\pi(y)k(x,y) \\
\end{aligned}
$$

- (b) If $\pi(y)q(y,x)=0$

$$
\begin{aligned}
\pi(x)k(x,y)
&=\pi(x)q(x,y)\min \left(1,\frac{\overbrace{\pi(y)q(y,x)}^{0}}{\pi(x)q(x,y)}\right)=0 \\
\pi(y)k(y,x)
&=\underbrace{\pi(y)q(y,x)}_{0}\min \left[1,\frac{\pi(x)q(x,y)}{\underbrace{\pi(y)q(y,x)}_{0}}\right]=0 \\
\end{aligned}
$$

- We still need irreducibility. How can we think about this?
- Let $\mathcal{X}$ be a nice bounded and connected region in $\mathbb{R}^k$.
- Suppose $\pi(x)>0\;\forall x\in \mathcal{X}$. If the proposal $q(x\to y)$
  satisfies $q(x\to y)>0$ and $q(y\to x)>0$, then all states are reachable from
  one another.
  {% marginfigure 'x-u-y' 'courses/stats-270/figures/lecture-12/x-u-y.png' 'Path from $x\to u\to y$.' %}
- Consider a possible path from $x\to y$.
  - Under $q$, this path has probability $q(x,u)\cdot q(u,y)>0$
  - Then the probability under MH chain for this path is $q(x,u)\min
    \left(1,\frac{\pi(u)q(u,x)}{\pi(x)q(x,u)}\right)\cdot q(u,y)\min \left(1,\frac{\pi(y)q(y,u)}{\pi(u)q(u,y)}\right) > 0$
  - Under the proposal distribution, the chain should be irreducible.
- Suppose $$S=\{x: \pi(x)>0\}=S_1\cup S_2$$, i.e. $S_1$ and $S_2$ are
  connected regions, but $S$ is not connected.
  {% marginfigure 'region-s' 'courses/stats-270/figures/lecture-12/region-s.png' 'Regions $S_1$ and $S_2$.' %}
- In this example, the proposal distribution has to be able to "jump" from one
  region to another, i.e. the "jumps" must be large enough.

## Gibbs Sampling

- Consider proposal move that involves changing one coordinate of $x$.
  $\vec{x}$ is $d$-dimensional: $\vec{x}=(x_1,\ldots,x_i,\ldots,x_d)$.
- Let $\vec{x}_i(y)=(x_1,\ldots,x_i=y,\ldots,x_d)$, i.e. changing the $i$th
  component to $y$.
- Let $\vec{x}_{-i}\in \mathbb{R}^{d-1}$, i.e. delete the $i$th component of $\vec{x}$.
- Construct $q(x,y)$ in two steps:
  1. Choose a coordinate $i$.
  2. Draw $$y\sim q_i(x_i\to y)$$. Set $$\vec{y}=\vec{x}_i(y)$$. This is
     $$q(x_i\to y\mid \vec{x_{-1}})$$.
- The MH ratio is then

$$
\begin{aligned}
\frac{\pi(\vec{y})q(\vec{y},\vec{x})}{\pi(\vec{x})q(\vec{x},\vec{y})}
&= \frac{\pi(\vec{x}_i(y))}{\pi(\vec{x})}\cdot \frac{q_i(y\to x_i)}{q_i(x_i\to y)} \\
\frac{\pi(\vec{x}_i(y))}{\pi(\vec{x})}
&= \frac{\pi_i(y\mid \vec{x}_{-i})\cdot \pi(\vec{x}_{-i})}{\pi_i(x_i\mid \vec{x}_{-i})\cdot \pi(\vec{x}_{-i})}
= \frac{\pi_i(y\mid \vec{x}_{-i})}{\pi_i(x_i\mid \vec{x}_{-i})} \\
\end{aligned}
$$

- This suggests that setting
  $$q_i(x_i\to y)=\pi_i(y\mid\vec{x}_{-1})$$ makes the ratio always 1, so
  you always accept the sample.
- **Gibbs sampling**: sample from $\pi_i(\cdot\mid \vec{x}_{-i})$ iteratively.

### Random Scan Gibbs

- $\vec{x}^{(t)}$ denotes the current value of $\vec{x}$.

1. Select $i$ from $$\{1,2,\ldots,d\}$$ randomly.
2. Draw $y$ from the conditional distribution $\pi_i(y\mid \vec{x}_{-i})$ and
   set $$\vec{x}_i^{(t+1)}(y)$$, $$\vec{x}_{-i}^{(t+1)}=\vec{x}_{-i}^{(t)}$$.

- **Proof**: This satisfies Detailed Balance because

$$
\begin{aligned}
\pi(\vec{x})k(\vec{x},\vec{y})
&=\pi(\vec{x})\cdot\frac{1}{d}\cdot\pi_i(y\mid\vec{x}_{-i}) \\
&=\frac{\pi(\vec{x})\pi(\vec{y})}{d\cdot\pi(\vec{x}_{-i})} \\
&=\frac{\pi(\vec{x})\pi(\vec{y})}{d\cdot\pi(\vec{y}_{-i})} \\
\end{aligned}
$$

- This operation is symmetric in $\vec{x}$ and $\vec{y}$.
- In practice, we don't use random scan, we use systematic scan, i.e. for
  $i=1,2,\ldots,d$, draw $$\vec{x}_i^{(t+1)}\sim \pi_i(\cdot\mid
  x_1^{(t+1)},\ldots,x_{i-1}^{(t+1)},x_{i+1}^{(t)},\ldots,x_d^{(t)})$$, i.e. you
  are always using the "latest" data.

### Example

- $$\vec{x}=(x_1,x_2)\sim \operatorname{Normal}\left(\begin{bmatrix}0 \\ 1\end{bmatrix},\begin{bmatrix}1 & \rho \\ \rho & 1\end{bmatrix}\right)$$, then
  - $x_1\mid x_2\sim \operatorname{Normal}\left(\rho x_2,1-\rho^2 \right)$
  - $x_2\mid x_1\sim \operatorname{Normal}\left(\rho x_1,1-\rho^2 \right)$
- Systematic scan is not a special case of MH because the following is not
  symmetric in $\vec{x}$ and $\vec{y}$. So, detailed balance is not satisfied in
  $\vec{x}\to\vec{y}$.

$$
\begin{aligned}
\pi(x)k(x,y)
&=\pi(x_1,x_2)\pi(y_1\mid x_2)\pi(y_2\mid y_1)
\end{aligned}
$$

- **Theorem**: If $\pi(\vec{x})>0$ for all $\vec{x}\in \mathbb{Z}^d$, then
  systematic scan Gibbs is a valid MCMC in the sense that the time average
  converges to the space average.
- **Proof**:
  1. Irreducibility $\pi(\cdot)>0$.
     {% marginfigure 'conditional-sampling' 'courses/stats-270/figures/lecture-12/conditional-sampling.png' 'Conditional sampling the system.' %}
  2. Each single coordinate update in the scan satisfies detailed balance. Each
     step leaves $\pi(\cdot)$ invariant. Hence the basic theorem of Markov Chain
     applies.

## Ising Model

$$
\begin{aligned}
x_i
&\in\{-1,1\} \\
\vec{x}
&=(x_1,\ldots,x_d) \\
\pi(\vec{x})
&\propto \exp\left(-\beta \sum_{i=1}^n x_i x_{i+1}\right) \\
\pi_i(x_i\mid \vec{x}_{-i})
&\propto \exp\left(\beta(x_1x_2 + x_2x_3+\cdots+x_{i-1}x_i+x_ix_{i+1}+x_{i+1}x_{i+2}\cdots\right) \\
&\propto \exp\left(\beta (x_{i-1}x_i + x_ix_{i+1})\right) \\
&= \exp\left(\beta x_i(x_{i-1}+x_{i+1})\right) \\
\pi_i(x_i\mid \vec{x}_{-i})
&= \frac{1}{Z_i}\exp\left(\beta x_i(x_{i-1}+x_{i+1})\right) \\
Z_i
&= \exp\left(\beta(x_{i-1}+x_{i+1})\right)+\exp\left(-\beta(x_{i-1}+x_{i+1})\right) \\
\end{aligned}
$$

- I.e. $x_i=1$ or $x_i=-1$.
- If $\pi(\vec{x})\propto \exp\left(-\sum_{c\in\mathcal{C}} v_c(x_c)\right)$
  where $\mathcal{C}$ is a set of local neighborhoods.
  {% marginfigure 'ising-neighborhoods' 'courses/stats-270/figures/lecture-12/ising-neighborhoods.png' 'Ising neighborhoods.' %}

### Efficiency

$$
\begin{aligned}
X&\sim \operatorname{Normal}\left(\begin{bmatrix}0 \\ 0 \end{bmatrix},\begin{bmatrix}1 &
\rho \\ \rho & 1\end{bmatrix}\right) \\
\operatorname{var}\left[X\right]
&= 1-\rho^2 \\
\text{step size}
&= \sqrt{1-\rho^2} \\
\end{aligned}
$$

- Sometimes a Markov chain can satisfy the theorem but it will take too long to simulate.
  {% marginfigure 'rho-steps' 'courses/stats-270/figures/lecture-12/rho-steps.png' 'Small steps with high $\rho$.' %}
