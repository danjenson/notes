---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 5: Sufficiency (2022-10-11)"
toc: true
---

# Sufficiency principle

- Book: "Theoretical Statistics" by Cox & Hinkley (Chapter 2)
- Sufficient statistics:
  - $Y\in\mathcal{Y}$: $y$ is distributed according to a density
    $f\in\mathcal{F}$
  - Let $S$ be a statistic, i.e. $S=S(Y): \mathcal{Y}\to\mathcal{S}$
- **Definition**: $S$ is a sufficient statistic if the conditional distribution
  of $Y$ given $S$ is the same for all $f\in\mathcal{F}$ where $\mathcal{F}$ is
  _any_ family of densities; there does not need to be a particular parametric
  family.
  - $f_{Y\mid S}(y\mid s)$
  - $g_{Y\mid S}(y\mid s)$
  - $\forall f,g\in\mathcal{F}\;f_{Y\mid S}(y\mid s)=g_{Y\mid S}(y\mid s)$ if
    $S$ is a sufficient statistic
- Example 1:

  - $$Z=\begin{bmatrix}x_i \\ y_i\end{bmatrix}$$ for $i=1,\ldots,n$ are iid
    vectors with $$\operatorname{Normal}\left(\begin{bmatrix} 0 \\
  0\end{bmatrix},\begin{bmatrix}1 & \rho \\ \rho & 1\end{bmatrix}\right)$$ density:
    $p(x,y)=\frac{1}{2\pi\sqrt{1-p^2}}\exp\left(-\frac{1}{2(1-p^2)}(x^2+y^2-2\rho xy)\right)$,
  - $$
    f(z)=\left(\frac{1}{2\pi\sqrt{1-\rho^2}}\right)\exp
    \left(-\frac{1}{2}\cdot\frac{1}{1-\rho^2}\left(\underbrace{\sum_{i=1}^n x_i^2 +
    \sum_{i=1}^n y_i^2}_{s_2} - 2\rho \underbrace{\sum_{i=1}^n x_iy_i}_{s_1}\right)\right)
    $$
  - Note that in the following, when integrating over the region where the
    sufficient statistics equal $s_1$ and $s_2$, $f_{Z\mid S}(z\mid s)$ is constant. To
    verify this, simply look at the preceding likelihood above.

  $$
  \begin{aligned}
  f_{Z\mid S}(z\mid s)
  &=\frac{f_Z(z)}{\int_{\{S_1(z)=s_1,S_2(z)=s_2\}}f_Z(z)\dd z} \\
  &=\frac{1}{\int_{\{S_1(z)=s_1,S_2(z)=s_2\}}1\dd z} \\
  &=\frac{1}{\operatorname{Area}(\{S_1(z)=s_1,S_2(z)=s_2\})}
  \end{aligned}
  $$

  - $f_{Z\mid S}=f_Z$ since $Z$ must be compatible with $S$ in order for the
    density to be non-zero. In other words, $Z$ is a more detailed event of $S$.
  - **This does not depend on $\rho$, so $S$ is sufficient.**
  - If $$\begin{bmatrix}x_i \\ y_i\end{bmatrix}\sim
  \operatorname{Normal}\left(\begin{bmatrix}\mu_1 \\
  \mu_2\end{bmatrix},\begin{bmatrix}1 & \rho \\ \rho & 1\end{bmatrix}\right)$$
    then the sufficient statistic would be:

$$
S=\left(\bar{X},\bar{Y},\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y}),\sum_{i=1}^n
(x_i-\bar{x})^2 + \sum_{i=1}^n (y_i-\bar{y})^2\right)
$$

- Example 2:

  - Suppose $\mathcal{F}=\\{f_1,f_2,\ldots,f_k\\}$; the sample space, $\mathcal{Y}$, is arbitrary,
    then there exists a $k-1$ dimensional sufficient statistic.
  - To construct it, let $\bar{f}(x)=\frac{1}{k}\sum_{i=1}^k f_i(x)$, without
    loss of generality, we can assume $\bar{f}(x)>0\;\forall x\in\mathcal{Y}$
    because that would imply that all densities for that $x$ are 0, so that $x$
    point can be excluded from the sample space.
  - Define $S_1(y)=\frac{f_1(y)}{\bar{f}(y)}$ and
    $S_k(y)=\frac{f_k(y)}{\bar{f}_k(y)}$. So, assume $\bar{f}(x)>0\;\forall
    x\in\mathcal{Y}$.
  - $k\bar{f}(y)=\sum_{j=1}^k f_j(y)=\sum_{j=1}^k S_j(y)\bar{f}(y)\implies
  \sum_{j=1}^k S_j(y)=k$
    - This means that $S$ is a $k-1$ dimensional statistic.
  - We claim that $S$ is a sufficient statistic.
  - Proof:

    - $$
      \mathcal{Y}_s=\{y\in\mathcal{Y}: S_j(y)=s_j, j=1,\ldots,k\}=\{y:
      f_i(y)=s_j\bar{f}(y), j=1,\ldots,k\}
      $$
    - Let $A\subset\mathcal{Y}_s$, then we can show it no longer depends on
      $f_j$, i.e. it no longer depends on which density you pick:

      $$
      \begin{aligned}
        \Pr_{f_j}(Y\in A\mid S=s)
        &=\Pr_{f_j}(Y\in A\mid Y\in \mathcal{Y}_s)
        \\ &=\frac{\int_A f_j(y)\dd y}{\int_{\mathcal{Y}_s}f_j(y)\dd y}
        \\ &=\frac{\int_A s_j\bar{f}(y)\dd y}{\int_{\mathcal{Y}_s}s_j\bar{f}(y)\dd y}
        \\ &=\frac{\int_A \bar{f}(y)\dd y}{\int_{\mathcal{Y}_s}\bar{f}(y)\dd y}
      \end{aligned}
      $$

    - Because it doesn't matter what $s_j$ is, $S$ is sufficient.

- Numerical example:
  {% marginfigure 'line-partition' 'courses/stats-270/figures/lecture-5/line-partition.png' 'Line partitioned by $S$' %}

  $$
  \begin{aligned}
  \mathcal{Y}&=[0,1]
  \\ \mathcal{F}&=\{f_1,f_2,f_3\}
  \\ f_1&=1\;\forall y\in [0,1]
  \\ f_2&=\begin{cases}1/2 & y\in[0,1/2] \\ 3/2 & y\in[1/2,1]\end{cases}
  \\ f_3&=\begin{cases}1/4 & y\in[0,1/4] \\ 5/4 & y\in[1/4,1]\end{cases}
  \\ \bar{f}&=\begin{cases}14/24 & y\in [0,1/4] \\ 22/24 & y\in [1/4,1/2]
  \\ 30/24 & y\in[1/2,1]\end{cases}
  \\ S&=\begin{bmatrix}f_1 / \bar{f} \\ f_2 / \bar{f}\end{bmatrix}\text{ (3rd is
   determined by first 2)}
   \\ S_1&=\begin{bmatrix}12/7 \\ 6/7\end{bmatrix}
   \quad S_2=\begin{bmatrix}12/11 \\ 6/11\end{bmatrix}
   \quad S_3=\begin{bmatrix}12/15 \\ 6/15\end{bmatrix}
   \\ &\text{Also sufficient:}
   \\ S_1&=\begin{bmatrix}0 \\ 0\end{bmatrix}
   \quad S_2=\begin{bmatrix}0 \\ 1\end{bmatrix}
   \quad S_3=\begin{bmatrix}1 \\ 0\end{bmatrix}
   \\ &\text{Another sufficient:}
  \\ S_1&=1\quad S_2=2\quad S_3=3
  \end{aligned}
  $$

- Observing $S$ allows us to pick out one of the subintervals.
- Once you pick out a value of $S_j$, you've identified a subinterval. And
  within that subinterval, each of the densities is different, but uniform. In
  other words, the conditional density is the same for each density given the
  sufficient statistic.
- Under any $f_j(\cdot)$, $Y$ is uniformly distributed within that
  subinterval.
- What is important about the sufficient statistic is its ability to
  partition, not the actual value. This shows that the sufficient statistic need
  not be unique.
- Sufficiency depends on the partition of the sample space, and not on the
  statistic itself.
  {% marginfigure 'y-partition' 'courses/stats-270/figures/lecture-5/y-partition.png'
  'Partition of sample space.' %}
- $\mathcal{Y}=\cup_{s\in\mathcal{S}}\mathcal{Y}_s$ is the partition of the
  sample space induced by a statistic $S$.
- For $S$ to be sufficient, all $f\in\mathcal{F}$ must have the same conditional
  distribution within each slice of $\mathcal{Y}_s$.
- The concept of sufficient and likelihood were developed by R.A. Fisher in the
  early 20th century. He gave the following theorem for the verification of
  sufficiency in parametric families.
- **Sufficiency Factorization Theorem**: If $Y\in f_\theta(\cdot)$ and
  $\theta\in\Omega$, then $S(Y)$ is sufficient iff $f_\theta(y)=g(s,\theta)h(y)$
  for some $g(\cdot,\cdot)$ and $h(\cdot)$.
- **Sufficiency Principle**: If $S$ is sufficient, then inference on the
  parameter $\theta$ should depend only on $S(Y)$ and not on any other aspects
  of $Y$. In other words, if we have $S(y_1)=S(y_2)$ (two possible realizations
  of the data), then inference on $\theta$ based on $Y=y_1$ should be identical to
  the inference based on $Y=y_2$.
- To consider why this is a reasonable principle, consider drawing $Y$ from
  $f_\theta(\cdot)$ in two stages.
  1. Generate $S=s$ from $f_{S;\theta}(s)$
  2. Generate $Y=y$ from $f_{Y\mid S;\theta}(y\mid s)$
- Then, $Y\sim f_\theta(\cdot)$
- If $S$ is sufficient, then the 2nd stage of the experiment contains no
  information on $\theta$ because $f_{Y\mid S;\theta}(y\mid s)$ does not depend
  on $\theta$.

# Bayesian Inference Automatically Satisfies the Sufficiency Principle

- Here, $y$ gives no new information about $\theta$:

$$
p(\theta\mid y)=\frac{\pi(\theta)f_{S;\theta}(s)f_{Y\mid S}(y\mid s)}{\int\pi(\theta)f_{S;\theta}(s)f_{Y\mid S}(y\mid s)\dd \theta}
=\frac{\pi(\theta)f_{S;\theta}(s)}{\int\pi(\theta)f_{S;\theta}(s)\dd \theta}
$$

# Minimal Sufficiency

- If $T$ is a statistic, and $S=g(T(Y))$ then the partition of $T$ is a
  refinement of that of $S$; that is, $S$ induces a coarser partition.
  {% marginfigure 'coarseness' 'courses/stats-270/figures/lecture-5/coarseness.png' '$S$
  is coarser than $T$.' %}
- If both $T$ and $S$ are sufficient, you should use $S$, since it is coarser.
- What if $S$ is not a function of $T$, i.e. they are not nested?
- Bahadur (1954) showed that in general there exists a unique coarsest
  sufficient partition. The corresponding statistic is the "minimal sufficient
  statistic".
