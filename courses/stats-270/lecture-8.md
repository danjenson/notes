---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 8: Something"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}{\operatorname{var}}
\newcommand{\sd}{\operatorname{sd}}
\newcommand{\cov}{\operatorname{cov}}
$$

- Example:

$$
\begin{aligned}
\begin{bmatrix}x \\ y\end{bmatrix}
&\sim \operatorname{Normal}\left(\begin{bmatrix}\mu_1 \\ \mu_2\end{bmatrix},\tau\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}\right) \\
\theta
&= (\tau, \mu_1,\mu_2) \\
\tau
&= \sigma^2 \\
f_\theta(x,y)
&= \frac{1}{2\pi\tau}\exp\left(-\frac{1}{2\tau}\left[(x-\mu_1)^2+(y-\mu_2)^2\right]\right) \\
\ell
&=\log f_\theta(x,y)=c - \log\tau-\frac{1}{2\tau}\left[(x-\mu_1)^2+(y-\mu_2)^2\right] \\
\dot\ell_1
&=\pdv{\ell}{\tau}=-\frac{1}{\tau}+\frac{1}{2\tau}\left[\left(\frac{x-\mu_1}{\sigma}\right)^2+\left(\frac{y-\mu_2}{\sigma}\right)^2\right] \\
\dot\ell_2
&=\pdv{\ell}{\mu_1}=\frac{1}{\sqrt{\tau}}\cdot\frac{x-\mu_1}{\sigma} \\
\dot\ell_3
&=\pdv{\ell}{\mu_2}=\frac{1}{\sqrt{\tau}}\cdot\frac{y-\mu_2}{\sigma} \\
\mathbb{E}\left[\dot\ell\right]
&= \vec{0} \\
c
&= \frac{2}{4}\var{Z^2}\text{ where }Z\sim \operatorname{Normal}\left(0,1\right) \\
\var{\dot\ell_1}
&=\frac{c}{\tau^2} \\
\var{\dot\ell_2}
&= \frac{1}{\tau} \\
\var{\dot\ell_3}
&= \frac{1}{\tau} \\
\cov(\dot\ell_1,\dot\ell_2)
&=
\end{aligned}
$$

- TODO finish
- If we have $$\vec{X}=\{(X_i,Y_i)\}_{i=1}^n$$, then what is the posterior?

$$
\begin{aligned}
p(\tau,\mu_1,\mu_1\mid\vec{x})&\propto\frac{1}{\tau^2}\cdot\frac{1}{\tau^n}\exp\left(-\frac{1}{2\tau}\left[\underbrace{\sum_{i=1}^n(x_i-\bar{x})^2+\sum_{i=1}^n(y_i-\bar{y})}_{S}+n(\bar{x}-\mu_1)^2+n(\bar{y}-\mu_2)^2\right]\right) \\
p(\tau\mid\vec{x})&\propto\frac{1}{\tau^{n+2}}\exp\left(-\frac{1}{2\tau}S\right)\underbrace{\int_{-\infty}^\infty
\exp\left(-\frac{n}{2\tau}(\bar{x}-\mu_1)^2\right)\dd
\mu_1}_{\sqrt{2\pi\tau/n}}\underbrace{\int_{-\infty}^\infty
\exp\left(-\frac{n}{2\tau}(\bar{y}-\mu_2)\right)^2\dd \mu_2}_{\sqrt{2\pi\tau/n}} \\
p(\tau\mid\vec{x})&\propto\frac{1}{\tau^{n+1}}\exp\left(-\frac{1}{2\tau}S\right)
\end{aligned}
$$

- $S$ is the corrected sum of squares.
- Let $\gamma=\frac{S}{\tau}$, i.e. $\tau=\frac{S}{\gamma}$
- This is a scaled inverse $\chi^2$ with $2n$ degrees of freedom.
- $p(\gamma\mid\vec{x})\propto
  \frac{1}{\tau^{n+1}}\exp\left(-\frac{1}{2}\gamma\cdot\frac{S}{\gamma^2}\right)\propto \gamma^{n-1}\exp\left(-\frac{1}{2}\gamma\right)$
  - This is $\operatorname{Gamma}\left(n,\frac{1}{2}\right)=\chi_{2n}^2$.
- Thus, conditional on $\vec{x}$, $\gamma=\frac{S}{\tau}\sim\chi_{2n}^2$.
- But conditional on $\vec{\theta}$, $\gamma\sim\chi_{2n-2}^2$.
- TODO finish
- Let us also compute the marginal posterior for the location parameters.

$$
\begin{aligned}
p(\tau,\mu_1,\mu_2\mid\vec{x})
&\propto\frac{1}{\tau^{n+2}}\exp\left(-\frac{1}{2\tau}\left[S+n(\bar{x}-\mu_1)^2-n(\bar{y}+\mu_2)^2\right]\right) \\
p(\tau,\mu_1\mid\vec{x})
\propto\frac{1}{\tau^{n+\frac{3}{2}}}\exp\left(-\frac{1}{2\tau}\left[S+n(\bar{x}-\mu_1)^2\right]\right) \\
\end{aligned}
$$

- To integrate $\tau$, recall the gamma function $\Gamma(\alpha)=\int_0^\infty
  x^{\alpha-1}e^{-x}\dd x=\int_0^\infty\beta^\alpha x^{\alpha-1}e^{-\beta x}\dd
  x$ (TODO CHECK)
- So, set $x=\frac{1}{y}$ (gives a Jacobian of $\frac{1}{y^2}$), then

$$
\begin{aligned}
\int_0^\infty\frac{1}{y^{\alpha+1}}\exp\left(-\frac{\beta}{y}\right)\dd
y=\frac{\Gamma(\alpha)}{\beta^\alpha}
\end{aligned}
$$

- Using this with $$\begin{cases} \alpha+1=n+\frac{3}{2} \\
  \beta=S+n(\bar{x}-\mu_1)^2\end{cases}$$
- Then we have

  $$
  \begin{aligned}
  p(\mu_1\mid\vec{x})
  &\propto\frac{\Gamma(\alpha)}{\beta^\alpha} \\
  &\propto\frac{1}{\left[S+n(\mu_1-\bar{x})^2\right]^{n+\frac{1}{2}}} \\
  &\propto\frac{1}{[1+\frac{1}{2n}\cdot\frac{n(\mu_1-\bar{x})^2}{S/2n}]^{n+\frac{1}{2}}}
  \end{aligned}
  $$

- TODO finish
- On the other hand, if we condition on $\theta$, then
- TODO finish this
- Example 2:
- TODO Write up
- We know that $n$ is the correct degrees of freedom.
- The problem is caused by the unbounded parameter space and the "improper"
  nature of Jeffrey's prior. You can't put a uniform prior over an unbounded
  space.
- In Example 1, $$\Omega=\left\{(\tau,\mu_1,\mu_2):\tau>0, \mu_1\in \mathbb{R}^1,\mu_2\in \mathbb{R}^1\right\}$$
- Approximate $\Omega$ by $$\Omega^{(k)}=$$
- Jeffrey's way of dealing with the problem.

  - Starting with any prior $\vec{\pi}(\theta)$, let $n\to\infty$, we have that
    $\sqrt{n}(\theta-\hat{\theta})\mid X_n\implies \operatorname{Normal}\left(0,i^{-1}(\theta)\right)$.

    - Now, if $i(\theta)$ is diagonal, $$\begin{bmatrix} i_{1,1}(\theta) & 0 \\
    0 & i_{2,2}(\theta)\end{bmatrix}$$ in this sense $\theta_1$ and $\theta_2$
      are "independent" parameters, then
      $\pi(\tau,\mu_1,\mu_2)=\pi(\tau)\pi(\mu_1)\pi(\mu_2)$ where

      $$
      \begin{aligned}
      \pi(\tau)
      &\propto i_{1,1}^{\frac{1}{2}}(\theta)\propto\frac{1}{\tau} \\
      \pi(\mu_1)
      &\propto i_{2,2}^{\frac{1}{2}}(\theta)\propto\frac{1}{\tau}\propto 1 \\
      \pi(\mu_1)
      &\propto i_{2,2}^{\frac{1}{2}}(\theta)\propto\frac{1}{\tau}\propto 1 \\
      \end{aligned}
      $$
