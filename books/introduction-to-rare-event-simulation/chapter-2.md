---
title: Introduction to Rare Event Simulation
title_url: "."
subtitle: "Chapter 2: Stochastic Models"
toc: true
---

# 2.1 Gaussian Processes

- Method to create correlated Gaussians with autoregressive moving average
  (ARMA(p,q)) structure p.17.

# 2.2 Markov Processes

# 2.3 Markov Chain Monte Carlo

- One runs an MCMC chain and samples it from time to time.
- If the chain has reached equilibrium, we can think of the samples as being
  from $\pi$, the stationary distribution.
- If you wait sufficient time between samples, you can view the samples as
  independent.
- Overview of Metropolis algorithm.
- Review of Barker's algorithm.
- Review of Gibb's sampler on $\Lambda$-valued process.
  - $\Lambda$-valued processes are usually colors, gray levels, spin states,
    etc.

## 2.3.1 Simulation of Markov Random Fields
