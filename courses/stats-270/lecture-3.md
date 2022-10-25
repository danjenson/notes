---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 3: Decision Theory (2022-10-04)"
---

- Review of statistical decision theory.
- When given a loss function, $L(a,\theta):\mathcal{A}\times\Omega\to
  \mathbb{R}^+$, nature chooses the $\theta\in \Omega$, while the statistician chooses an
  action, $a\in \mathcal{A}$, based on an observation $x\sim \Pr_\theta(\cdot)$
  according to a decision rule $\delta(x):\mathcal{X}\to\mathcal{A}$.
- For the Bayesian with prior $\pi(\cdot)$, the solution is easy:

$$
\delta^*(x)=\arg\min_{a\in\mathcal{A}}\mathbb{E}\left[L(a,\Theta)\mid X=x\right]
$$

- Example 1:

  $$
   \begin{aligned}
     p(x\mid\theta)&=\begin{cases}1/\theta & \text{if }0\le x\le \theta \\ 0
     &\text{otherwise}\end{cases} \\
     \pi(\theta)&=\begin{cases}\theta e^{-\theta}&\text{if }\theta>0 \\ 0
     &\text{otherwise}\end{cases} \\
     p(x,\theta)&=\begin{cases}e^{-\theta}&\text{if }0\le x\le \theta \\
     0&\text{otherwise}\end{cases} \\
     p(x)&=\int_x^\infty e^{-\theta}\dd\theta = e^{-x} \\
     p(\theta\mid x)&=e^{-(\theta-x)}
     \end{aligned}
  $$

- Decision here: let $\mathcal{A}=\Omega$, i.e. the action means it is our
  estimate of $\theta$. Then, use squared error loss, so
  $L(a,\theta)=(a-\theta)^2$. So, you want to calculate
  $\mathbb{E}\left[L(a,\Theta)\mid x\right]=\int_x^\infty(a-\theta)^2
  e^{x-\theta}\dd\theta$. If you take the derivative with respect to $a$ and
  then set it equal to 0.

$$
\begin{aligned}
0&=\dv{a}\int_{x}^\infty (a-\theta)^2e^{x-\theta}\dd\theta \\
&=\int_x^\theta 2(a-\theta)e^{x-\theta}\dd\theta \\
a&=\frac{\int_x^\infty\theta \exp(x-\theta)\dd\theta}{\int_x^\infty
\exp(x-\theta)\dd\theta} \\
&=x+1 \\
\delta^*(x)&=x+1
\end{aligned}
$$

- More generally, if the loss function is $(\theta-a)^2$, then the Bayes
  decision rule is $\delta^\*(x)=\mathbb{E}\left[\Theta\mid X=x\right]$. The
  proof is that $\mathbb{E}\left[L\mid
  x\right]=\mathbb{E}\left[(\Theta-a)^2\right]$ when $\Theta\sim$posterior
  distribution given $x$. This is not true for other loss functions.

- Example 2: Same as Example 1 but $X=(Y_1,\ldots,Y_10)$ where $Y_i$ are iid.
- Example 3: Now $\theta\in\\{1,2,3\\}$, L=0,1,or 4 depending on $\Theta$ and
  $a$.
- Example 4: $X\sim \operatorname{Binomial}\left(3,\theta\right)$ and
  $\theta\in\left\\{\frac{1}{4},\frac{3}{4}\right\\}$. This implies that
  $L=0$ when $a=\theta$ and $L=\frac{1}{4}$ when $a\ne 0$.

| Example | $\Omega$                                   | $\mathcal{A}$                              | $\mathcal{X}$     | $\mathcal{D}$                                                                        |
| ------- | ------------------------------------------ | ------------------------------------------ | ----------------- | ------------------------------------------------------------------------------------ |
| 1       | $(0,\infty)$                               | $(0,\infty)$                               | $(0,\infty)$      | $\\{\delta: \mathbb{R}^+\to \mathbb{R}^+\\}$                                         |
| 2       | $(0,\infty)$                               | $(0,\infty)$                               | $(0,\infty)^{10}$ | $\\{\delta: (\mathbb{R}^+)^{10}\to \mathbb{R}^+\\}$                                  |
| 3       | $\\{1,2,3\\}$                              | $\\{1,2,3\\}$                              | $(0,\infty)^{10}$ | $\\{\delta: (\mathbb{R}^+)^{10}\to \\{1,2,3\\}\\}$                                   |
| 4       | $\left\\{\frac{1}{4},\frac{3}{4}\right\\}$ | $\left\\{\frac{1}{4},\frac{3}{4}\right\\}$ | $\\{0,1,2,3\\}$   | $\left\\{\delta: \\{0,1,2,3\\}\to \left\\{\frac{1}{4},\frac{3}{4}\right\\}\right\\}$ |

- Decisions are not so easy for the frequentist. The performance of
  $\delta(\cdot)$ is judged by the risk function (i.e. the frequentist expected
  loss).

$$
r^\delta(\theta)=\mathbb{E}\left[L(\delta(X),\theta)\right]=\int
L(\delta(x),\theta)\Pr_\theta(x)\dd x
$$

- Example 1:
  - $r^\delta(\theta)=\int_0^\theta(\delta(x)-\theta)^2\frac{1}{\theta}\dd
  x=\int_0^1(\delta(\theta y)-\theta)^2\dd y$
  - Consider the following decision rules which can be drawn:
  - $\delta_1(x)=x+1$: $r^{\delta_1}(\theta)=\int_0^1(\theta
  y+1-\theta)^2\dd y
  y=\frac{1}{3}\theta^2+(1-\theta)$
  - $\delta_2(x)=\frac{1}{2}$: $r^{\delta_2}(\theta)=\int_0^1(\frac{1}{2}-\theta)^2\dd y=\left(\frac{1}{2}-\theta\right)^2$
  - $\delta_3(x)=-\frac{1}{2}$: $r^{\delta_3}(\theta)=\int_0^1(-\frac{1}{2}-\theta)^2\dd y=\left(-\frac{1}{2}-\theta\right)^2$
  - $\delta_4(x)=2x+1$: $r^{\delta_4}(\theta)=\frac{1}{3}\theta^2+1$
- A rule $\delta_1$ is said to be as good as $\delta_2$ if
  $r^{\delta_1}(\theta)\le r^{\delta_2}(\theta)\;\forall\theta\in\Omega$.
  Furthermore, if the inequality is strict at some $\theta$, then $\delta_1$ is
  better than $\delta_2$.
- A rule is **admissible** if there does not exist any rule better than it.
  - Frequentists argue to only use admissible rules. However, you may need to
    choose from among many admissible rules using heuristics like minimax.
- **Theorem**: Suppose $\Omega$ is either finite or compact in $\mathbb{R}^k$.
  In the latter case, also assume that $r^\delta(\cdot)$ is continuous in
  $\theta$, for all $\delta$. If $\delta^\*$ is a Bayes rule with respect to
  $\pi(\cdot)$. satisfying the following, then $\delta^\*$ is admissible:
  1. Support of the prior $\pi(\cdot)$ is the whole $\Omega$.
  2. $\int \pi(\theta)r^{\delta^\*}(\theta)\dd\theta < \infty$.
- **Proof**: First note that $E(L)=E(E(L\mid X))=E(E(L\mid\Theta))$, then:

  $$
  \begin{aligned}
  &\int \pi(\theta)r^{\delta^*}(\theta)\dd\theta \\
  &=\int \pi(\theta)
  \mathbb{E}\left[L(\delta^*(X),\Theta)\mid\Theta=\theta\right]\dd\theta \\
  &= \mathbb{E}\left[L(\delta^*(X),\Theta)\right] \\
  &\le \int p(x) \mathbb{E}\left[L(\delta(x),\Theta)\mid X=x\right]\dd x \\
  &= \int\pi(\theta)r^\delta(\theta)\dd\theta
  \end{aligned}
  $$

- Thus, $\int
  \pi(\theta)\left[r^{\delta^\*}(\theta)-r^\delta(\theta)\right]\dd\theta \le 0$
- The claim is is that if $\delta$ is better than $\delta^\*$, then we will get
  a contradition.
- **Proof**: If $\delta$ is better than $\delta^\*$, then there exists a
  measurable portion of the space, $B$ such that:

$$
\int_B\pi(\theta)\left[r^{\delta^*}(\theta)-r^\delta(\theta)\right]\dd\theta
+ \int_{B^c}\pi(\theta)\left[r^{\delta^*}(\theta)-r^\delta(\theta)\right]\dd\theta \ge \epsilon \Pr_\pi(B) > 0
$$
