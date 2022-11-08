---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 8: Jeffrey's Prior and the Score Function (2022-10-20)"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

# Example 1

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
\mathbb{E}\left[\dot\ell_1\right]
&= -\frac{1}{\tau} + \frac{1}{2\tau}\left(\mathbb{E}\left[\chi^2(1)\right] + \mathbb{E}\left[\chi^2(1)\right]\right)=-\frac{1}{\tau}+\frac{1}{2\tau}\cdot (1+1) = 0 \\
\mathbb{E}\left[\dot\ell_2\right]
&= \frac{1}{\sqrt{\tau}}\cdot\frac{\mathbb{E}\left[X\right]-\mu_1}{\sigma}=0 \\
\mathbb{E}\left[\dot\ell_3\right]
&= \frac{1}{\sqrt{\tau}}\cdot\frac{\mathbb{E}\left[Y\right]-\mu_2}{\sigma}=0 \\
\mathbb{E}\left[\dot\ell\right]
&= \vec{0} \\
\var{\dot\ell_1}
&=\frac{c}{\tau^2}
\text{ where }c=\frac{2}{4}\var{Z^2}\text{ and }Z\sim \operatorname{Normal}\left(0,1\right) \\
\var{\dot\ell_2}
&= \frac{1}{\tau} \\
\var{\dot\ell_3}
&= \frac{1}{\tau} \\
\cov{\dot\ell_1}{\dot\ell_2}
&=\cov{\frac{1}{2\tau}\left(\frac{x-\mu_1}{\sigma}\right)^2}{\frac{1}{\sqrt{\tau}}\left(\frac{x-\mu_1}{\sigma}\right)} \\
&=\frac{1}{2\tau^{\frac{3}{2}}}\mathbb{E}\left[Z^3\right]=0 \\
\end{aligned}
$$

- Recall that $\dot\ell$ is the score function.
- The variance of $\dot\ell_1$:
  - The first term is constant so can be ignored.
  - The second term is squared under variance.
  - Because $X$ and $Y$ are independent, their variances are additive.
  - Each squared term is a standard normal, $Z$, and there are 2 of them.
  - Putting this together you get $\frac{2}{4\tau^2}\cdot\var{Z^2}$.
- The covariance is a linear operation, and since here $X$ and $Y$ are
  independent, the covariance of the terms involving them is 0.

$$
\begin{aligned}
\cov{X}{X+Y}
&=\mathbb{E}\left[X(X+Y)\right]-\mathbb{E}\left[X\right]\mathbb{E}\left[X+Y\right] \\
&=\mathbb{E}\left[X^2\right] + \mathbb{E}\left[XY\right]-\mathbb{E}\left[X\right]\mathbb{E}\left[X\right] - \mathbb{E}\left[X\right]\mathbb{E}\left[Y\right] \\
&=\mathbb{E}\left[X^2\right] - \mathbb{E}^2\left[X\right] + \mathbb{E}\left[XY\right]-\mathbb{E}\left[X\right]\mathbb{E}\left[Y\right] \\
&= \var{X} + \cov{X}{Y}
\end{aligned}
$$

- Furthermore, because the standard normal is centered around 0 and the first
  moment is 0, all subsequent moments are also 0. Thus, $\mathbb{E}\left[Z^3\right]=0$.
- Similarly, the other covariances are also 0.
- This is Jeffrey's prior, and suggest that the should be the square root of the
  determinant of the following matrix or $\pi(\theta)\propto \frac{1}{\tau^2}$
  (the determinant is the product of the diagonal, then take the square root):

$$
\begin{aligned}
i(\theta)&\propto \begin{bmatrix}
  \frac{1}{\tau^2} & 0 & 0 \\
  0 & \frac{1}{\tau} & 0 \\
  0 & 0 & \frac{1}{\tau} \\
\end{bmatrix}
\end{aligned}
$$

- If we have $$\vec{X}=\{(X_i,Y_i)\}_{i=1}^n$$, then what is the posterior? {%
  sidenote 'assume' 'Assumes $\mu_1,\mu_2$ are uniform on $[-\infty,\infty]$' %}

$$
\begin{aligned}
p(\tau,\mu_1,\mu_1\mid\vec{x})&\propto\underbrace{\frac{1}{\tau^2}}_{\text{Jeffrey's Prior}}\cdot\underbrace{\frac{1}{\tau^n}\exp\left(-\frac{1}{2\tau}\left[\sum_{i=1}^n(x_i-\mu_1)^2+\sum_{i=1}^n(y_i-\mu_2)\right]\right)}_{\text{likelihood}} \\
p(\tau,\mu_1,\mu_1\mid\vec{x})&\propto\frac{1}{\tau^2}\cdot\frac{1}{\tau^n}\exp\left(-\frac{1}{2\tau}\left[\underbrace{\sum_{i=1}^n(x_i-\bar{x})^2+\sum_{i=1}^n(y_i-\bar{y})^2}_{S\text{ (corrected sum of squares)}}+n(\bar{x}-\mu_1)^2+n(\bar{y}-\mu_2)^2\right]\right) \\
p(\tau\mid\vec{x})&\propto\frac{1}{\tau^{n+2}}\exp\left(-\frac{1}{2\tau}S\right)\underbrace{\int_{-\infty}^\infty
\exp\left(-\frac{n}{2\tau}(\bar{x}-\mu_1)^2\right)\dd
\mu_1}_{\sqrt{2\pi\tau/n}}\underbrace{\int_{-\infty}^\infty
\exp\left(-\frac{n}{2\tau}(\bar{y}-\mu_2)\right)^2\dd \mu_2}_{\sqrt{2\pi\tau/n}} \\
p(\tau\mid\vec{x})&\propto\frac{1}{\tau^{n+2}}\exp\left(-\frac{1}{2\tau}S\right)\cdot\frac{2\pi\tau}{n} \\
p(\tau\mid\vec{x})&\propto\frac{1}{\tau^{n+1}}\exp\left(-\frac{1}{2\tau}S\right)
\end{aligned}
$$

- The trick above comes from the analysis of variance:

$$
\begin{aligned}
&\sum_{i=1}^n (x_i-\mu)^2 \\
&= \sum_{i=1}^n (x_i^2-2x_i\mu + \mu_2) \\
&= \sum_{i=1}^n (x_i^2) - 2n\bar{x}\mu + n\mu^2 \\
&= \sum_{i=1}^n (x_i^2) - 2n\bar{x}\mu + n\mu^2 + 2n\bar{x} - 2n\bar{x} \\
&= \sum_{i=1}^n (x_i^2) - 2n\bar{x}^2 + n\bar{x}^2  + n\mu_2 - 2n\bar{x}\mu + n\bar{x}^2  \\
&= \sum_{i=1}^n (x_i^2) - 2(n\bar{x})\bar{x} + n\bar{x}^2  + n(\mu^2 - 2\bar{x}\mu + \bar{x}^2)  \\
&= \sum_{i=1}^n (x_i^2 - 2x_i\bar{x} + \bar{x}^2)  + n(\mu^2 -\bar{x})^2  \\
&= \sum_{i=1}^n (x_i - \bar{x})^2  + n(\mu^2 -\bar{x})^2 \\
\end{aligned}
$$

- Let $r=\frac{S}{\tau}$, i.e. $\tau=\frac{S}{r}$, and Jacobian is $\frac{S}{r^2}$.
- This is a scaled inverse $\chi^2$ with $2n$ degrees of freedom.
- $p(r\mid\vec{x})\propto
  \frac{1}{\tau^{n+1}}\exp\left(-\frac{1}{2}r\right)\cdot\frac{S}{r^2}\propto r^{n-1}\exp\left(-\frac{1}{2}r\right)\sim\operatorname{Gamma}\left(n,\frac{1}{2}\right)=\chi_{2n}^2$.
- Thus, conditional on $\vec{x}$, $r=\frac{S}{\tau}\sim\chi_{2n}^2$
  (scale-inverse $\chi^2$).
- But conditional on $\vec{\theta}$, $r\sim\chi_{2n-2}^2$ (when you correct for
  the mean, you lose 2 degrees of freedom).
- This suggests that the degrees of freedom in the posterior derived from
  Jeffrey's prior may be incorrect. On the other hand, if we use a prior which
  is not Jeffrey's prior, i.e. $\pi(\theta)\propto\frac{1}{\tau}$, then the
  degrees of freedom is $2n-2$.
- Let us also compute the marginal posterior for the location parameters.

$$
\begin{aligned}
p(\tau,\mu_1,\mu_2\mid\vec{x})
&\propto\frac{1}{\tau^{n+2}}\exp\left(-\frac{1}{2\tau}\left[S+n(\bar{x}-\mu_1)^2+n(\bar{y}-\mu_2)^2\right]\right) \\
p(\tau,\mu_1\mid\vec{x})
&\propto\frac{1}{\tau^{n+\frac{3}{2}}}\exp\left(-\frac{1}{2\tau}\left[S+n(\bar{x}-\mu_1)^2\right]\right) \\
\end{aligned}
$$

- To integrate $\tau$, recall the gamma function $\Gamma(\alpha)=\int_0^\infty
  x^{\alpha-1}e^{-x}\dd x=\int_0^\infty\beta^\alpha x^{\alpha-1}e^{-\beta x}\dd
  x$
  - This is derived by substituting $x=\beta x$, which creates the Jacobian
    $\beta$ multiplied by the other term $\beta^{\alpha-1}$.
- So, set $x=\frac{1}{y}$ (gives a Jacobian of $\frac{1}{y^2}$), then

$$
\begin{aligned}
\Gamma(\alpha)
&=\int_0^\infty\frac{1}{y^2}\cdot\beta^\alpha\frac{1}{y^{\alpha-1}}\exp\left(-\frac{\beta}{y}\right)\dd y \\
\frac{\Gamma(\alpha)}{\beta^\alpha}
&=\int_0^\infty\frac{1}{y^{\alpha+1}}\exp\left(-\frac{\beta}{y}\right)\dd y \\
\end{aligned}
$$

- Using this with $$\begin{cases} \alpha+1=n+\frac{3}{2}\text{, i.e.
  }\alpha=n+\frac{1}{2} \\
  \beta=S+n(\bar{x}-\mu_1)^2 \\
  y=\tau \\
  \end{cases}$$
- Then we have

  $$
  \begin{aligned}
  p(\mu_1\mid\vec{x})
  &\propto\frac{\Gamma(\alpha)}{\beta^\alpha} \\
  &\propto\frac{1}{\left[S+n(\bar{x}-\mu_1)^2\right]^{n+\frac{1}{2}}} \\
  &\propto\frac{1}{S^{n+\frac{1}{2}}[1+\frac{1}{2n}\cdot\frac{n(\bar{x}-\mu_1)^2}{S/2n}]^{n+\frac{1}{2}}} \\
  &\propto\frac{1}{[1+\frac{1}{2n}\cdot\frac{n(\bar{x}-\mu_1)^2}{S/2n}]^{n+\frac{1}{2}}}
  \end{aligned}
  $$

- If you compare this to the t-student distribution:

$$\frac{1}{\left(1+\frac{t^2}{\nu}\right)^{\frac{\nu+1}{2}}}$$

- Then, if you set $t=\frac{\sqrt{n}(\bar{x}-\mu_1)}{\sqrt{S/2n}}$ and $\nu=2n$,
  then $\frac{\sqrt{n}(\bar{x}-\mu_1)}{\sqrt{S/2n}}\sim t_{2n}$.
  - So, the marginal posterior of $\mu_1$ is a t-density with $2n$ degrees of
    freedom, which we suspect is wrong; it should be $2n-2$).
- On the other hand, if we condition on $\theta$, then

$$
\begin{aligned}
\frac{\sqrt{n}(\bar{x}-\mu_1)}{\sqrt{\frac{S}{2n-2}}}
&=\frac{\sqrt{n}\left(\frac{\bar{x}-\mu_1}{\sigma}\right)}{\sqrt{\frac{S}{\sigma^2}\cdot\frac{1}{2n-2}}}
\sim\frac{\operatorname{Normal}\left(0,1\right)}{\sqrt{\frac{\chi_{2n-2}^2}{2n-2}}}
=t_{2n-2}
\end{aligned}
$$

- Thus, curiously, we arrive at

$$
\begin{aligned}
\frac{\sqrt{n}(\bar{x}-\mu_1)}{\sqrt{S}}
&=\begin{cases}
\frac{1}{\sqrt{2n}}t_{2n}&\text{conditioning on }\vec{x} \\
\frac{1}{\sqrt{2n-2}}t_{2n-2}&\text{conditioning on }\vec{\theta}
\end{cases}
\end{aligned}
$$

- **Claim**: $2n-2$ is the correct value.

# Example 2

$$
\begin{aligned}
\begin{bmatrix}X_i \\ Y_i \end{bmatrix}
&\sim \operatorname{Normal}\left(\begin{bmatrix}\mu_i \\ \mu_i\end{bmatrix},\tau\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}\right) \\
\theta
&= (\tau,\mu_1,\mu_2,\ldots,\mu_n) \\
S
&= \sum_{i=1}^n\left[(x_i-\hat{\mu}_i)^2+(y_i-\hat{\mu}_i)^2\right]
= \frac{1}{2}\sum_{i=1}^n (x_i-y_i) \\
\hat{\mu}_i
&= \frac{X_i+Y_i}{2} \\
R
&= \frac{S}{\tau} \\
\text{Jeffrey's Prior}
=
i(\theta)&\propto \begin{bmatrix}
  \frac{1}{\tau^2} & 0 & 0 & 0 \\
  0 & \frac{1}{\tau} & 0 & 0 \\
  0 & 0 & \ddots & \vdots \\
  0 & 0 & \cdots &  \frac{1}{\tau} \\
\end{bmatrix} \\
\pi(\theta)
&\propto\frac{1}{\tau^{\frac{2+n}{2}}}
\end{aligned}
$$

- Note that $$\sum_{i=1}^n\left[(x_i-\hat{\mu}_i)^2+(y_i-\hat{\mu}_i)^2\right]
=\frac{1}{2}\sum_{i=1}^n (x_i-y_i)$$ becomes a normal distribution which can be
  used to derive $R\mid\theta\sim\chi_{n}^2$.

- The same computation gives:

$$
\begin{aligned}
R\mid\vec{x}
&\sim\chi_{2n}^2 \\
R\mid\vec{\theta}
&\sim \chi_n^2
\end{aligned}
$$

- So, $$\frac{1}{n}\cdot R=\frac{1}{n}\cdot\frac{S}{\tau}\to\begin{cases}2&\text{conditional on
  }\vec{x} \\ 1&\text{conditional on }\vec{\theta}\end{cases}$$

- We know that $n$ is the correct degrees of freedom. So, using Jeffrey's prior
  asymptotically leads to an incorrect result, 2.
- The problem is caused by the unbounded parameter space and the "improper"
  nature of Jeffrey's prior. You can't put a uniform prior over an unbounded
  space.
- In Example 1, $$\Omega=\left\{(\tau,\mu_1,\mu_2):\tau>0, \mu_1\in \mathbb{R}^1,\mu_2\in \mathbb{R}^1\right\}$$
- Approximate $\Omega$ by $$\Omega^{(k)}=\left\{\frac{1}{k}<\tau<k:
  -k\sigma\le\mu_1\le k\sigma,\;-k\sigma\le\mu_2\le k\sigma\right\}$$
- Jeffrey's prior on $\Omega^{(k)}\propto\frac{1}{\tau^2}$
- Marginal for $\tau$:
  $\pi(\tau)=\int_{-k\sigma}^{k\sigma}\int_{-k\sigma}^{k\sigma}\frac{1}{\tau^2}\dd\mu_1\dd\mu_2=\frac{(2k\sigma)^2}{\tau^2}\propto\frac{1}{\tau}$, which is **correct**.
- On the other hand, if we set $$\Omega^{(k)}=\left\{\frac{1}{k}<\tau<k:
  -k\le\mu_1\le k,\;-k\le\mu_2\le k\right\}$$, then by the same calculation we
  get $\pi(\tau)\propto\frac{1}{\tau^2}$, which is **wrong**.

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
