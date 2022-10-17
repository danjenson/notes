---
title: Introduction to Rare Event Simulation
title_url: "."
subtitle: "Chapter 1: Random Number Generation"
toc: true
---

# 1.1 Uniform Generators

- A random number generator is nothing more than a deterministic algorithm
  operating on a finite state machine (the computer) that produces numbers with
  certain distributional properties.
- The most popular generator is the **congruential generator**:
  - $M>0$ is a large prime number called the modulus.
  - $0 < a < M$ is the multiplier.
  - $C$ is the increment.
  - $0 < c_0 < M$ is the initial value or seed.

$$
c_k=[ac_{k-1}+C\mod (M)]
$$

- Typically, numbers are converted to something on the real interval with
  $u_k=c_k/M$.
- A rule of thumb for linear generators in general is that the usable sample
  size is close to $\sqrt{P}$ where $P$ is the period of the generator.
- Random generators may look normal but then have correlations in various
  ranges.

{% marginfigure 'bit-correlation'
'books/introduction-to-rare-event-simulation/figures/chapter-1/last-8-bits.png' 'Correlations among bigs.' %}

- Various generators such as the KISS generator and Mother of all RNGs.

# 1.2 Nonuniform Generation

## 1.2.1 The Inversion Method

$$
F^{-1}(u)=\inf\{x: F(x)\ge u\}
$$

- Since the CDF of a distribution is uniform, invert the CDF (when possible),
  simulate a uniform random variable and plug it into the inverse.
- **Lemma 1.2.1**: For $0<u<1,u\le F(x)$ if and only if $F^{-1}(u)\le x$.
- Several examples, including the Box-Muller method for generating two random
  normals.

## 1.2.2 The Acceptance-Rejection Method

1. Find a dominating density $g$ over the support of $f$ such that $cg$ (where
   $c$ does not depend on $x$) is greater than the maximum of $f$.
2. Simulate an $X\sim g$.
3. Simulate a uniform from $[0, cg(x)]$. If this value is below $f(x)$, accept
   it. In other words, accept if $u < f(x)/cg(x)$.

- Note that $g$ must have heavier tails and sharper infinite peaks than $f$.

# 1.3 Discrete Distributions

- Inversion can be used for discrete distributions as well.

$$
F(X-1) = \sum_{i<X}p_i<U\le\sum_{i\le X}p_i=F(X)
$$

- You can do a sequential search; example with Poisson on p.14.

## 1.3.1 Inversion by Truncation of a Continuous Analog

- You can use a continuous distribution $G$ as a dominating density for discrete
  distribution $F$ by using the floor function.

## 1.3.2 Acceptance-Rejection

- Acceptance-rejection can also be used on discrete distributions.
- **Hybrid Rejection Algorithm**:
  1. Generate $Y\sim g$, Set $X\leftarrow \lfloor Y\rfloor$
  2. Generate a uniform $U$ on $[0,1]$.
  3. Accept and return $X$ if $Ucg(Y)\le p(X)$, otherwise repeat.
