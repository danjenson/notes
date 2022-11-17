---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 14: Gradient & Hamiltonian Monte Carlo Moves (2022-11-15)"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

## Gradient Moves

- $f(x)=\frac{1}{c}\exp\left(-U(x)\right)$ where $x\in \mathbb{R}^d$ and $U(x)$
  is called the potential energy.
- Equivalent to negative log likelihood up to a constant.
- Gradient moves:
  - Given the current value of $x$
  - Let $g(x)=\nabla\log f(x)=-\nabla U(x)$
  - Move to $$y=x+\underbrace{\delta g(x)}_{\text{gradient shift of step size
  }\delta} +\underbrace{\epsilon Z}_{\text{random walk with step size
  }\epsilon}$$ where $Z\sim \operatorname{Normal}\left(0,I_d\right)$
- Langevin move:
  - $\epsilon=\sqrt{2\delta}$ or $\delta=\frac{1}{2}\epsilon^2$
  - As $\delta\to 0$, then this approximates the evolution of a stochastic
    differential equation (SDE) which gives $f(\cdot)$ as its equilibrium
    distribution. This is true without the rejection step.
- Potential difficulty under MH:

  - In a region where $|g(x)|$ is large.
  - We want to take a reasonably large step in $g(x)$ direction.
  - You're unlikely to step back with the random walk when the gradient is
    large.
  - This means that the reverse probability is small, making the acceptance
    probability very small.
  - If you use a large Gaussian, then you are ignoring the gradient information.
  - There is a tension between reversibility for detailed balance and wanting to
    use the gradient to get to high probability regions.
  - E.g. $$U(x)=-\frac{1}{2}x^2,\delta g(x)+\epsilon Z=-\delta x +\epsilon Z$$
    - For acceptance, you need $$\delta|x|\approx\sqrt{\delta}\implies
    \delta\sim\frac{1}{|x|^2}$$ (a small step size).
      {% marginfigure 'gradient-gaussian' 'courses/stats-270/figures/lecture-14/gradient-gaussian.png' 'Gradient+Gaussian steps.' %}

- One idea is to use a reversible gradient move.
  - $$y=x+\delta z_0 g(x)+\epsilon Z$$
  - $z_0$ is either 1 or 0 with $0.5$ probability each.
  - This enables you to jump forward and backward with equal probability.
    {% marginfigure 'reversible-gradient' 'courses/stats-270/figures/lecture-14/reversible-gradient.png' 'Reversible-Gradient steps.' %}

## Hamiltonian Moves (HMC)

- Radford, Neal, "MCMC Using Hamiltonian Dynamics"
- Want to sample from the Boltzmann distribution.
- In statistics, $T=1$, $U(q)=-\log$posterior density.
- In physics, $T$ is the temperature of the system and $U$ is the potential
  energy, and $Z_T$ is called the partition function (a function of $T$).

$$
\begin{aligned}
p(q)
&=\frac{1}{Z_T}\exp\left(-U(q)/T\right) \\
q
&=\begin{bmatrix}q_1\\ \vdots \\ q_d\end{bmatrix}
\end{aligned}
$$

- How is this related to dynamics?
- Recall that in classical mechanics, force is given by $-\nabla U(q)$.
- Newtons equations of motion: $m \frac{d^2 q(t)}{dt^2}=-\nabla U(q)$.
  - 2nd order differential equation.
- Hamilton reformulated this as a first order system.
- Let:

$$
\begin{aligned}
p
&=\begin{bmatrix}p_1\\ \vdots \\ p_d\end{bmatrix}=\text{momentum vector} \\
k(p)
&=\frac{1}{2}\sum_{i=1}^d \frac{p_i^2}{m_i}=\text{kinetic energy} \\
H(q, p)
&= U(q)+K(p)=\text{Hamiltonian function or total energy}
\end{aligned}
$$

- This implies:

$$
\begin{aligned}
\dv{q_i}{f}
&=\frac{1}{m_i}p_i \\
\dv{p_i}{f}
&= -\pdv{U}{q_i} \\
\end{aligned}
$$

- In 1816-1898, Boltzmann argued that in the long run, the relative frequency in
  phase $$\Omega=\{(q,p)\}$$, when marginalized to $\vec{q}$, it is described by
  $p(q)\propto \exp\left(-\frac{U(q)}{T}\right)$. This is an ergodic hypothesis.
  (G.D. Birhoff 1931 gave a proof.)
- In fact, Birhoff's work suggested that, in thermal equilibrium at temperature
  $T$, the joint distribution of $(q, p)$ is given by

$$
\begin{aligned}
p(q,p)
&=\frac{1}{c}\exp\left(-\frac{1}{T}H(q,p)\right) \\
&=\underbrace{\left(\frac{1}{Z_T}\exp\left(\frac{-U(q)}{T}\right)\right)}_{\text{potential energy}}\underbrace{\left(\frac{1}{c}\exp\left(-\frac{1}{T}\sum_{i=1}^d \frac{p_i^2}{2m_i}\right)\right)}_{\text{kinetic energy}} \\
p_i&\sim \text{independent}\operatorname{Normal}\left(0,m_i T\right) \\
\vec{q}&\sim \operatorname{Boltzman dist}\propto \exp\left(\frac{-U(q)}{T}\right) \\
\end{aligned}
$$

- We mimic this by a Markov Chain with HMC proposal.
- Assume for simplicity:
  - $T=1$
  - Current state is $(q, p)$
- Then:

  1. Use $(q^{(0)}, p^{(0)})$, $p_i^{(0)}\sim
     \operatorname{Normal}\left(0,m_i\right)$ for $i=1,\ldots,d$.
  2. Use $(q^{(0), p^{(0)}})$ as an initial value, and evolve the system using
     the Hamiltonian dynamic for a duration $\ell$, to get $(q^{(1)},p^{(1)})$.
  3. Propose to move to a new point $(q^{(1)},-p^{(1)})$. Accept with
     probability:

     $$
     \begin{aligned}
      r
     &=\min\left[1,
     \frac{\exp\left(-H(q^{(1)},-p^{(1)})\right)}{\exp\left(-H(p^{(0)},q^{(0)})\right)}\right]
     \end{aligned}
     $$

- To implement step 2, we divide $[0, \ell]$ into steps of size $\epsilon$.
- Use "leapfrog" algorithm as follows
  $\left(t=0,\epsilon,2\epsilon,\ldots,\frac{\ell}{\epsilon}\right)$:

  - Given $(q(t),p(t))$

  $$
  \begin{aligned}
  p_i \left(t+\frac{\epsilon}{2}\right)
  &=p_i(t)-\left(\epsilon\right)\cdot\pdv{U}{q_i}(q(t)) \\
  q_i \left(t+\frac{\epsilon}{2}\right)
  &=q_i(t)-\left(\epsilon\right)\cdot\pdv{p_i\left(t+\frac{\epsilon}{2}\right)}{m_i} \\
  p_i \left(t+\epsilon\right)
  &=p_i(t+\frac{\epsilon}{2})-\left(\frac{\epsilon}{2}\right)\cdot\pdv{U}{q_i}(q(t+\epsilon)) \\
  \end{aligned}
  $$

  - Repeat this for $L=\\frac{\ell}{\epsilon}$ steps.

- **Theorem**: HMC transition with "leapfrog" implementation leaves
  $p(q,p)=\frac{1}{c}\exp\left(-H(q,p)\right)$ invariant.
- **Proof**: Define a transformation $(p^{(0)},q^{(0)})\to_S(q^{(1)},-p^{(1)})$
  1. $S$ is 1-1 in $\Omega$.
  2. $S$ preserves volume because it is a composition of "shear"
     transformations.
     {% marginfigure 'shear' 'courses/stats-270/figures/lecture-14/shear.png' 'HMC shear transformations.' %}

{% fullwidth 'courses/stats-270/figures/lecture-14/detailed-balance.png'
'Detailed balance.' %}
