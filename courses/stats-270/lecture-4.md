---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 4: Decision Theory continued..."
toc: true
---

# Overview

- Example:
  - $\Pr_\theta(x)=\frac{1}{\theta}\mathbb{I}_{[0,\theta]}(x)$
  - $\theta\in\Omega=\\{\theta_1,\theta_2\\}=\\{1,2\\}$
  - Observe $x=(y_1,y_2,y_3)$ where $y_i$ are iid $\sim\Pr_\theta(\cdot)$
  - Let $\delta:[0,2]^3\to\mathcal{A}=\\{1,2\\}$
  - The risk function $$r^\delta(\theta)=\begin{bmatrix}r^\delta(\theta_1) \\r^\delta(\theta_2)\end{bmatrix}$$
  - The Loss is $$L=\begin{cases} 0&\text{ if }\delta=\theta \\ 1&\text{ if }\delta\ne\theta\end{cases}$$
  - If you assume that in position 1, $\theta_1$ is the correct value, then the
    loss is the probability of selecting $\theta_2$.
    $$r^\delta=\begin{bmatrix}p_1(\delta=2) \\ p_2(\delta=1)\end{bmatrix}$$
  - Now, choose a decision rule:
    - $\delta_a$: always choose $\theta_1$, then $$r=\begin{bmatrix}0 \\ 1\end{bmatrix}$$
    - $\delta_b$: always choose $\theta_2$, then $$r=\begin{bmatrix}1 \\ 0\end{bmatrix}$$
    - $$
      \delta_c=\begin{cases}1&\text{if } \max(y_1,y_2)\le 0.9\\
      2&\text{otherwise}\end{cases}$$ then $$r=\begin{bmatrix}p_1(\max > 0.9) \\ p_2(\max\le 0.9)\end{bmatrix}=\begin{bmatrix}1-0.9^2\\ \left(\frac{1}{2}\right)^2(0.9)^2\end{bmatrix}=\begin{bmatrix}0.19 \\ 0.2025\end{bmatrix}
      $$
    - $$
      \delta_d=\begin{cases}1&\text{if }y\le 0.4048\\
      2&\text{otherwise}\end{cases}
      $$
      which implies $$r=\begin{bmatrix}0.5942 \\ 0.2024\end{bmatrix}$$
- Generally, if $\Omega=\\{\theta_1,\ldots,\theta_k\\}$ then $r^\delta(\cdot)$
  is represented by a point in k-dimensional space:

  $$
  \vec{r}^\delta=\begin{bmatrix}r^\delta(\theta_1)\\ \vdots \\
  r^\delta(\theta_k)\end{bmatrix}
  $$

- Let $\mathcal{D}$ be the set of possible decisions.
- Randomized decisions: let $\delta_1,\ldots,\delta_n$ are decision rules. Let
  $\delta^\*=\delta_z$ where $$x=\begin{cases}1\\ \vdots \\ m\end{cases}$$ with
  probability $$\begin{bmatrix}\alpha_1 \\ \vdots \\ \alpha_m\end{bmatrix}$$
  - The risk for a randomized rule is
    $$
    r^{\delta^*}(\theta)=\mathbb{E}_\theta\left[L(\delta_z,\theta)\right]=\mathbb{E}_\theta\left[\sum_{i=1}^m
    \alpha_i L(\delta_i,\theta)\right]=\sum_{i=1}^m \alpha_i
    r^{\delta_i}(\theta)
    $$
- The set of all randomized rules is a convex hull.
  {% marginfigure 'lecture-4-convex-hull' 'courses/stats-270/figures/lecture-4/convex-hull.png' 'Convex hull of decision points.' %}

- Let $S$ be the set of risk points of randomized rules: $$S=\{y\in \mathbb{R}^k: y=r^{\delta^*} \text{ for some randomized rule}\}$$
- **Lemma**: $S$ is a convex set in $\mathbb{R}^k$.

# Admissibility and the risk set

- An admissible rule is optimal in taht it cannot be improved across all
  $\theta\in\Omega$.
- The lower quadrant of $\vec{x}$: $Q_\vec{x}=\\{\vec{y}\in \mathbb{R}^k: y_i\le x_j,\; j=1,\ldots,k\\}$
- If $\vec{y}\ne \vec{x}$ where $\vec{x}=\vec{r}^\delta$,
  $\vec{y}=\vec{r}^{\delta'}$ then $\delta'$ is better than $\delta$ if and only
  if $y\in Q_\vec{x}$.
- **Lemma**: A decision rule is admissible iff its risk point $\vec{x}$
  satisfies $S\cap Q_\vec{x}=\\{\vec{x}\\}$.
  {% marginfigure 'lecture-4-lower-quadrant' 'courses/stats-270/figures/lecture-4/lower-quadrant.png' 'Lower quadrant admissibility.' %}

# Bayes rules and the risk set

- Let $\pi(\cdot)$ be the prior.

$$
\vec{\pi}=\begin{bmatrix}\pi(\theta_1)\\ \vdots \\
\pi(\theta_k)\end{bmatrix}=\begin{bmatrix}\pi_1 \\ \vdots \\
\pi{k}\end{bmatrix}
$$

such that $\pi_i\ge0\;\forall i$ and $\sum_i \pi_i=1$.

- All decision rules with the same $\pi$-averaged risk must lie in a hyperplane:
  $H_b=\\{y: \sum_i \pi_iy_i=b\\}$.
  {% marginfigure 'lecture-4-pi-average-minimization' 'courses/stats-270/figures/lecture-4/pi-average-minimization.png' '$\pi$-average minimization.' %}
- If $r^\delta(\theta_j)=y_j$ then $\vec{y}$ is the risk point for $\delta$.
- $\pi$-averaged risk for $\delta$ is $\sum\_{i=1}^k \pi(\theta_i)y_i$.
- This suggests that we have shown that the Bayes rules minimize the
  $\pi$-averaged risk.
- To find the Bayes value with respect to $\pi(\cdot)$ we change $b$ so that
  $H_b$ becomes tangent to the risk set $S$.
- **Theorem**: If $\Omega$ is finite and $\mathcal{A}$ is finite, under some
  regularity conditions, then for any admissible rule $\delta$, there is a Bayes
  rule that is as good as $\delta$, i.e. you don't need to go outside of the
  Bayes rules. Proof:
  - Let $\delta$ be an admissible rule, and $x=r^\delta$, then $S\cap
      Q_x=\\{x\\}$ .
  - Let $$T=Q_x\setminus\{x\}$$, then $T$ is convex.
    {% marginfigure 'lecture-4-convex-T' 'courses/stats-270/figures/lecture-4/convex-T.png' '$T$ is a convex set.' %}
  - $S$ and $T$ are two disjoint convex sets (when you take out $x$).
  - By the Separating Hyperplane Theorem, $\exists\vec{\alpha}\ne 0$
    such that $\sum_{i=1}^k\alpha_i y_i\le\sum_{i=1}^k\alpha_j z_j$ if
    $\vec{y}\in T$ and $\vec{z}\in S$.
  - Claim that $\alpha_j\ge 0\;\forall j=1,\ldots,k$.
    - Suppose $\alpha_1<0$, then $\sum_{i=1}^k \alpha_i y_i=\alpha_1y_1+\ldots$
    - If we let $y_1\to -\infty$, then $\sum_{i=1}^k\alpha_i y_i\to\infty$. This
      is a contradiction because it is not $\le \sum_{j=1}^k\alpha_j z_j$. This
      contradicts the separating property of $H_b$.
  - We define $\pi_j=\frac{\alpha_j}{\sum_{i=1}^k\alpha_i}$; now, $\vec{\pi}$ is
    a probability vector. If $\delta$ achieves the minimum $\pi$-averaged risk and
    if the minimum is uniquely achieved, then $\delta$ is a Bayes rule.
    {% marginfigure 'lecture-4-min-pi-averaged-risk' 'courses/stats-270/figures/lecture-4/min-pi-averaged-risk.png' 'Bayes rules are the corners of the lower hull.' %}
  - Conditions:
    - Regularity condition.
    - Distribution of $x$ is continuous under all $\theta$.
  - When is the minimum not uniquely achieved?
