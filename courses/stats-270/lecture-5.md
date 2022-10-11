---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 5: Sufficiency"
---

- Sufficiency principle
- Book: "Theoretical Statistics" by Cox & Hinkley (Chapter 2)
- Sufficient statistics:
  - $Y\in\mathcal{Y}$: $y$ is distributed according to a density
    $f\in\mathcal{F}$
  - Let $S$ be a statistic, i.e. $S=S(Y): \mathcal{Y}\to\mathcal{S}$
- **Definition**: $S$ is a sufficient statistic if the conditional distribution of $Y$ given
  $S$ is the same for all $f\in\mathcal{F}$.
  - $f_{Y\mid S}(y\mid s)$
  - $g_{Y\mid S}(y\mid s)$
  - $\forall f,g\in\mathcal{F}\;f_{Y\mid S}(y\mid s)=g_{Y\mid S}(y\mid s)$ if
    $S$ is a sufficient statistic
- Example 1:
  - $$Z=\begin{bmatrix}x_i \\ y_i\end{bmatrix}$$ for $i=1,\ldots,n$ are iid
    vectors with density
    $p(x,y)=\frac{1}{2\pi\sqrt{1-p^2}}\exp\left(-\frac{1}{2(1-p^2)}(x^2+y^2-2\rho xy)\right)$,
    i.e. $$\operatorname{Normal}\left(\begin{bmatrix} 0 \\
  0\end{bmatrix},\begin{bmatrix}1 & p \\ p & 1\end{bmatrix}\right)$$
  - $$f(z)=$$
  - So $S$ is sufficient.
  - If $$\begin{bmatrix}x_i \\ y_i\end{bmatrix}\sim
  \operatorname{Normal}\left(\begin{bmatrix}\mu_1 \\
  \mu_2\end{bmatrix},\begin{bmatrix}1 & \rho \\ \rho & 1\end{bmatrix}\right)$$
  - TODO note here
- Example 2:
  - Suppose $\mathcal{F}=\\{f_1,f_2,\ldots,f_k\\}$; $\mathcal{Y}$ is arbitrary,
    then there exists a $k-1$ dimensional sufficient statistic to construct it,
    but $\bar{f}(x)=\frac{1}{k}\sum_{i=1}^k f_i(x)$, without loss of generality,
    we can assume $\bar{f}(x)>0\;\forall x\in\mathcal{Y}$.
  - Define $S_1(y)=\frac{f_1(y)}{\bar{f}(y)}$ and
    $S_k(y)=\frac{f_k(y)}{\bar{f}_k(y)}$.
  - $k\bar{f}(y)=\sum_{i=1}^k f_i(y)=\sum_{i=1}^k S_i(y)\bar{f}(y)\implies
  \sum_{i=1}^k S_i(y)=k$
  - We claim that $S$ is a sufficient statistic.
  - Proof:
    - $$
      \mathcal{Y}_s=\{y\in\mathcal{Y}: S_j(y)=s_j, j=1,\ldots,k\}=\{y:
      f_i(y)=s_j\bar{f}(y), j=1,\ldots,k
      $$
    - Let $A\subset \mathcal{Y}_s$ then $P_{f_i}(Y\in A\mid S=s)=P_{f_i}(Y\in A\mid Y\in
      \mathcal{Y}\_s)$
    - TODO integral
    - Test
