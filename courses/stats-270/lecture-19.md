---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 19: Doob's Theorem on Bayesian Consistency (2022-12-08)"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

## Doob's Theorem on Bayesian Consistency

- Suppose $Y_1,Y_2,\ldots$ are iid according to $p_\theta(\cdot)$ and
  $y\in\mathcal{Y}$, $\theta\in\Omega\in \mathbb{R}^k$ (a bounded rectangle)
- **Definition**: An estimator $T_n=t_n(Y_1,\ldots,Y_n)$ is a consistent
  estimator if $\forall \theta\in\Omega$, for any $\varepsilon>0$, we have
  $p_\theta(|T_n-\theta|>\varepsilon)\to 0$ as $n\to\infty$.
- Let $\pi(\cdot)$ be a prior density and $$T_n^*=\mathbb{E}\left[\Theta\mid
  Y_1,\ldots,Y_n\right]=t_n^*(Y_1,\ldots,Y_n)$$
- **Theorem (Doob)**: If there exists any consistent estimator, then the
  $$T_n^*$$ will satisfy the following:

  $$
  \begin{aligned}
  p_\theta(\lim_{n\to\infty}T_n^*=\theta)
  &=1 \quad\text{strong consistentcy}\\
  \end{aligned}
  $$

  - Except when $\theta$ is in a set of $\pi(\cdot)$ of probability 0.

- **Proof**:
  - **Prelude**:
    - Consider the probability space $(\Omega\times\mathcal{Y}^\infty, Q)$
    - Under $Q$, the sample point $(\theta, y_1,y_2,\ldots)$ is generated by
      drawing $\theta\sim\pi(\cdot)$, then $y_1,y_2,\ldots$ are iid from
      $p_\theta(\cdot)$
  - **Claim A**:
    - There exists a random variable $$T_\infty^*$$ on
      $$\Omega\times\mathcal{Y}^\infty$$ such that
      $$T_\infty^*=g(\theta,y_1,y_2,\ldots)$$, with
      $$\mathbb{E}_{Q}\left[T_\infty^*\right]<\infty$$ such that
      $$p_Q(T_n^*\to T_\infty^*)=1$$
    - This means that $$p_Q\left(t_n^*(Y_1,\ldots,Y_n)\to
    g(\Theta,Y_1,Y_2,\ldots)\right)=1$$
  - **Claim B**:
    - $$\mathbb{E}_{Q}\left[T_\infty^*-\Theta\right]^2=0$$
    - So, $$p_Q(T_\infty^*=\Theta)=1$$
- By Claim A, we can replace $$T_\infty^*$$ by $$\lim_{n\to\infty}T_n^*$$, then
  $$p_Q(\lim_{n\to\infty}T_n^*=\Theta)=1$$, which implies
  $$1-p_Q(\lim_{n\to\infty}T_n^*=\Theta)=0$$
- Thus,

$$
\begin{aligned}
\int_\Omega \pi(\theta)\dd\theta-\int_\Omega\pi(\theta)\underbrace{p_\theta(\lim_{n\to\infty}t_n^*(Y_1,\ldots,Y_n\mid\theta)}_{\lambda(\theta)}\dd\theta
&=\int_\Omega \pi(\theta)(1-\lambda(\theta))\dd\theta=0
\end{aligned}
$$

- This implies that $\lambda(\theta)=1$ unless $\pi(\theta)=0$
- So, $$p_\theta(\lim_{n\to\infty}T_n^*=\theta)=1$$ unless $\pi(\theta)=0$.
- Hence, $$T_n^*$$ is strongly consistent for almost all $\theta$ under the
  prior.
- $$T_\infty^*=g(\theta,Y_1,Y_2,\ldots)=\theta$$ except for $\theta$ in a
  $\pi$-null set.
- In order to prove Claim A, we need a result form probability theory.
- $\mathcal{F}_n$ is the $\sigma$-filed of events generated by
  $$X_1,X_2,\ldots,X_n$$
- Given $$(\Lambda,\mathcal{F},P)$$, let
  $$\mathcal{F}_1\subset\mathcal{F}_2\subset\mathcal{F}_3\cdots$$ be a sequences
  of sub $\sigma$-fields and $X_n$ is measurable with respect to
  $\mathcal{F}_n$, then $$\{(X_n,\mathcal{F}_n):n=1,2,\ldots\}$$ is a martingale
  if:
  1. $$\mathbb{E}\left[X_n\right]<\infty\;\forall n$$
  2. With probability 1,
     $$\mathbb{E}\left[X_{n+1}\mid\mathcal{F}_n\right]=X_n$$
- Martingale Convergence Theorem (Doob)
  - Let $X_1,X_2,\ldots$ be a martingale and $\mathbb{E}\left[X_n\right]\le
  k,\;\forall n$ then there exists a random variable $X$ with
    $\mathbb{E}\left[X\right]<k$ so that $X_n\to X$ with probability 1.
  - This means that for any $\lambda\in\Lambda$ in the sample space
    $X_1(\lambda),X_2(\lambda),\ldots$ is just a sequences of real numbers
  - In general, this sequence may or may not converge to a limit.
- To prove Claim A, start with $$(\Omega\times\mathcal{Y}^\infty,Q)$$
  - Let $$\mathcal{F}_n$$ be a $\sigma$-field generated by $Y_1,\ldots,Y_n$
  - These are clearly increasing: $$\mathcal{F}_1\subset\mathcal{F}_2\cdots$$
  - Furthermore,

$$
\begin{aligned}
\mathbb{E}\left[T_{n+1}^*\mid\mathcal{F}_n\right]=
&=\mathbb{E}\left[\mathbb{E}\left[\Theta\mid Y_1,\ldots,Y_n,Y_{n+1}\right]\mid Y_1,\ldots,Y_n\right] \\
&=\mathbb{E}\left[\Theta\mid Y_1,\ldots,Y_n\right] \\
&=T_n^* \\
\end{aligned}
$$

- So, the martingale convergence theorem applies, and there exists a random
  variable $$T_\infty^*$$ such that $$T_n^*\to T_\infty^*$$ with $Q$ probability
  1.
- Proof of Claim B that $$\mathbb{E}_{Q}\left[T_\infty^*-\Omega\right]^2=0$$

  $$
  \begin{aligned}
  \mathbb{E}_{Q}\left[T_n^*-\Theta\right]^2
  &= \mathbb{E}\left[\mathbb{E}\left[T_n^*-\Theta\right]^2\mid Y_1,\ldots,Y_n\right] \\
  &\le \mathbb{E}\left[\mathbb{E}\left[T_n-\Theta\right]^2\mid Y_1,\ldots,Y_n\right] \\
  &=\mathbb{E}_Q\left[T_n-\Theta\right]^2 \\
  &=\int_\Omega\pi(\theta)\underbrace{\mathbb{E}_{\theta}\left[T_n-\theta\right]^2}_{\text{by Claim C:}\to 0}\dd\theta
  \\
  \end{aligned}
  $$

- If $T_n$ is a consistent estimate guaranteed by the construction of the
  theorem.
- Claim C: $$\mathbb{E}_{\theta}\left[T_n-\theta\right]^2\to0$$
- Apply the Bounded Convergence Theorem

$$
\begin{aligned}
\mathbb{E}_{Q}\left[T_n^*-\Theta\right]^2
&\to \mathbb{E}_{Q}\left[T_\infty^*-\Theta\right]^2=0 \\
\end{aligned}
$$

- **Proof of Claim C**:

  - Want to prove $$\mathbb{E}_{\theta}\left[T_n-\theta\right]^2\to 0$$
  - Suppose $$|\Omega|<M$$
    $$
    \begin{aligned}
    \mathbb{E}_{\theta}\left[T_n-\theta\right]^2
    &\le\int_{|T_n-\theta|>\varepsilon}|T_n-\theta|^2\dd p_\theta(Y_1,\ldots,Y_n) +
      \int_{|T_n-\theta|<\varepsilon}|T_n-\theta|^2\dd p_\theta(Y_1,\ldots,Y_n)
      \\
    &\le 4M^2 \underbrace{p_\theta(|T_n-\theta|>\varepsilon)}_{\to 0\text{ as }n\to\infty} + 2\varepsilon^2 \\
    &\le 3\varepsilon^2 \\
    \end{aligned}
    $$
  - Thus, $$\mathbb{E}_{\theta}\left[T_n-\theta\right]^2\to 0$$.

- Two loose ends.

  1. We have consistency with prior probability 1.
  2. There has to exist a consistent estimator. How can you guarantee the
     existence of a consistent estimator?

- **Theorem**: Let $X_1,X_2,\ldots$ iid according to $f_\theta(\cdot)$ and
  $\theta\in\Omega$. If $\Omega$ is compact with respect to some metric,

  $$
  d(\theta,\theta')=\lVert f_\theta(\cdot)-f_{\theta'}(\cdot)\rVert=\int_\Omega
  |f_\theta(y)-f_{\theta'}(y)|\dd\mu(y)
  $$

  - Then there exists and estimator $T_n$ such that
    $d(T_n,\theta)\to_{p_\theta} 0$ for all $\theta\in\Omega$.
