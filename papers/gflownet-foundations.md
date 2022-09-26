---
title: GFlowNet Foundations
toc: true
---

## Prelude

**Question**: What classes of problem do GFlowNets address?

**Answer**: They calculate free energies, i.e. partition functions, and
associated distributions, including conditional and marginal distributions.

## Terminology

### Terms

- **GFN**: Generative Flow Network.
- **MCMC**: Markov Chain Monte Carlo.
- **PPO**: Proximal Policy Optimization.
- **RL**: Reinforcement Learning.
- **TD**: Temporal Difference.
- **Proxy**: A function that approximates an oracle, i.e. $R(x)$, trained using
  $(x,y)$ pairs, which can incorporate (Bayesian) uncertainty.
- **Active learning**: A context in which "student" and "teacher" interact during training.
- **Flow matching**: The total flow going into a state must match the total
  flow leaving the state, except for the source, $s_0$, and sink(s), $s_f$ or
  $s_T$.
- **DAG**: Directed Acyclic Graph, $G=(\mathcal{S},\mathbb{A})$, i.e. a
  directed graph in which no trajectory $\tau=(s_1,\ldots,s_n)$ such that
  $s_1=s_n$
- **Pointed DAG**: A DAG $G=(\mathcal{S},\mathbb{A})$ such that there exist two
  states $s_0,s_f\in\mathcal{S}$ that satisfy $\forall s\in\mathcal{S}\setminus
  \\{s_0\\}\ s_0 < s \text{ and }\forall
  s\in\mathcal{S}\setminus\\{s_f\\}\ s < s_f$, i.e. there are source, $s_0$,
  and sink (or final), $s\_f$, states.
- **Complete trajectory**: A trajectory in a pointed DAG that starts at $s_0$
  and ends in $s\_f$, i.e. $\tau=(s_0,s_1,\ldots,s_n,s_{n+1}=s_f)$.
- **Terminating state**: Any state that is a parent of the sink state, $s_f$,
  i.e. $\\{s : s\to s_f\in\mathbb{A}\\}$; a terminating state may have
  other children from the sink state
- **Terminating edge**: Any edge between a terminating state an the sink, i.e.
  $s\to s_f$.
- **Markovian flow**: A flow is Markovian if $P(s\to
  s'\mid\tau)=P(s\to s'\mid s)=P_F(s'\mid s)$ for any $s\neq s_0$,
  outgoing edge $s\to s'$, and trajectory
  $\tau=(s_0,s_1,\ldots,s_n=s)\in\mathcal{T}^{partial}$. A flow is
  non-Markovian if the flow can remember past history.
- **Energy function**: Maps a state to a real value, $\mathcal{E}:\mathcal{S}\to \mathbb{R}$.
- **Free energy**: $\mathcal{F}(s)$ such that
  $e^{-\mathcal{F}(s)}:=\sum_{s^{\prime}: s^{\prime} \geq
  s}e^{-\mathcal{E}\left(s^{\prime}\right)}$

### Notation

- $\mathcal{S}$: The set of states, i.e. the state space.
- $\mathbb{A}$: The subset of $\mathcal{S}\times\mathcal{S}$ representing edges
  or transitions, i.e. $s\to s'$.
- $\tau$: A trajectory $\tau=(s_1,\ldots,s_n)$ of elements of $\mathcal{S}$
  such that every transition $s_t\to s_{t+1}\in\mathbb{A}$ and $n>1$.
  Also represented as $\tau=\to\ldots\to s_n$.
- $s\in\tau$: State $s$ is in trajectory $\tau$, i.e. $\exists t\in\\{1,\ldots,
  n\\}\;s_t=s$.
- $s\to s'\in\tau$ means $\exists t\in \\{1,\ldots,n-1\\}\;
  s\_t=s,s_{t+1}=s'$.
- $\lvert\tau\rvert$: The length of a trajectory is the number of edges.
- $s < s'$: Strict partial order (irreflexive, asymmetric, and transitive)
  where $s$ comes before $s'$ in a trajectory.
- $s \le s'$: Partial order (reflexive, antisymmetric, and transitive) where
  $s$ comes before $s'$ in a trajectory.
- $s \lessgtr s'$: No order relation between $s$ and $s'$.
- $\operatorname{Par}(s)$: The parent set: $\\{s'\in\mathcal{S} : s'\to s\in
  \mathbb{A}\\}$ given a DAG $G=(\mathcal{S},\mathbb{A})$.
- $\operatorname{Child}(s)$: The child set: $\\{s'\in\mathcal{S} : s\to s'\in
  \mathbb{A}\\}$ given a DAG $G=(\mathcal{S},\mathbb{A})$.
- $\mathcal{T}$: The set of complete trajectories in a pointed DAG.
- $\mathcal{T}^{partial}$: The set of (possibly incomplete) trajectories in a
  pointed DAG.
- $\mathcal{T}_{s,f}\subseteq\mathcal{T}^{partial}$: The set of trajectories
  starting in $s$ and ending in $s_f$ where
  $s\in\mathcal{S}\setminus\\{s_f\\}$.
- $\mathcal{T}_{0,s}\subseteq\mathcal{T}^{partial}$: The set of trajectories
  starting in $s_0$ and ending in $s$ where
  $s\in\mathcal{S}\setminus\\{s_0\\}$.
- $\mathcal{T}_{s\to s',s_f}$: The set of trajectories starting with
  $s\to s'$ and ending in $s_f$.
- $\mathcal{T}_{0,s\to s'f}$: The set of trajectories starting with
  in $s_0$ and ending with $s\to s'$.
- $d_{s,f}$: The maximum trajectory length in $\mathcal{T}_{s,f}$.
- $d_{0,s'}$: The maximum trajectory length in $\mathcal{T}_{0,s}$.
- $\mathbb{A}^{-f}$: $\left\\{s \to s^{\prime} \in \mathbb{A},
  s^{\prime} \neq s_f\right\\}$, the set of non-terminating edges in $G$
- $\mathbb{A}^f$: $\left\\{s \to s^{\prime} \in \mathbb{A},
  s^{\prime}=s_f\right\\}=\mathbb{A} \backslash \mathbb{A}^{-f}$, the set of
  terminating edges in $G$,
- $\mathcal{S}^f$: $\left\\{s \in \mathcal{S}, s \to s_f \in
  \mathbb{A}^f\right\\}=\operatorname{Par}\left(s_f\right)$, the set of
  terminating states in $G$.
- $F(\tau)$: A non-negative function $F:\mathcal{T}\mapsto\mathbb{R}^+$ defined
  on the set of **complete** trajectories $\mathcal{T}$. $F$ induces a measure
  over the $\sigma$-algebra $\Sigma=2^\mathcal{T}$, the power set on the set of
  complete trajectories $\mathcal{T}$.
- $(\mathcal{T}, 2^\mathcal{T}, F)$: A measure space where $F$ denotes both a
  function of complete trajectories and its corresponding measure over
  $(\mathcal{T},2^\mathcal{T})$.
- $(G,F)$: A flow network where $G$ is a pointed DAG and $F$ is a trajectory
  flow.
- $F(s)\coloneqq
  F(\\{\tau\in\mathcal{T}:s\in\tau\\})=\sum_{\tau\in\mathcal{T}:s\in\tau}F(\tau)$ :
  The flow through a state $F:\mathcal{S}\mapsto\mathbb{R}^+$ is the measure of
  the set of complete trajectories going through that state.
- $F(s\to s')\coloneqq F(\\{\tau\in\mathcal{T}: s\to
  s'\in\tau\\})=\sum_{\tau\in\mathcal{T}:s\to s'\in\tau}F(\tau)$: The
  flow through an edge $F:\mathbb{A}\mapsto\mathbb{R}^+$ is the measure of the
  set of complete trajectories going through a particular edge.
- $F(s\to s_f)$: A terminating flow.
- $\mathcal{F}(G)$: the set of flows on pointed DAG $G$, i.e. the set of
  functions from $\mathcal{T}$, the set of complete trajectories in $G$, to
  $\mathbb{R}^+$.
- $\mathcal{F}_{Markov}(G)$: the set of flows in $\mathcal{F}(G)$ that are
  Markovian for pointed DAG $G$.
- $Z\coloneqq F(\mathcal{T})=\sum_{\tau\in\mathcal{T}}F(\tau)$: The total flow,
  i.e. the sum of the flows of all complete trajectories.
- $P(A)\coloneqq\frac{F(A)}{F(\mathcal{T})}=\frac{F(A)}{Z}$: The flow
  probability is the measure $P$ over the measurable space
  $(\mathcal{T},2^\mathcal{T})$ associated with $F$ where $\forall
  A\subseteq\mathcal{T}$.
- $P(A\mid B)\coloneqq\frac{F(A\cap B)}{F(B)}$ where $\forall
  A,B\subseteq\mathcal{T}$.
- $P(s)\coloneqq\frac{F(s)}{Z}$: The probability of going through a state. This
  does not correspond to a distribution over states; namely,
  $\sum_{s\in\mathcal{S}}P(s)\neq 1$.
- $P(s\to s')\coloneqq\frac{F(s\to s')}{Z}=P_B(s\mid
  s')P(s')=P_F(s'\mid s)P(s)$: The probability of going through an edge.
- $P(\tau)\coloneqq\frac{F(\tau)}{Z}$: The probability of a trajectory.
- $P_T(s)\coloneqq P(s\to s_f)=\frac{F(s\to s_f)}{Z}$:
  Terminating state probability. Unlike $P(s)$, $P_T(s)$ is well-defined; i.e.
  $P_T(s)\ge 0\;\forall s\in \mathcal{S}^f$ and $\sum_{s\in\mathcal{S}^f}P_T(s)=1$.
- $P\_F(s'\mid s)\coloneqq P(s\to s'\mid s)=\frac{F(s\to
  s')}{F(s)}$: The forward transition probability; it satisfies $\forall
  s\in\mathcal{S}\setminus\\{s_f\\},\; \sum\_{s'\in
  \operatorname{Child}(s)}P\_F(s'\mid s)=1$
- $P\_B(s\mid s')\coloneqq P(s\to s'\mid s')=\frac{F(s\to
  s')}{F(s')}$: The backward probability function defined on $\mathbb{A}$; it
  satisfies $\forall s\in\mathcal{S}\setminus\\{s_0\\},\; \sum\_{s'\in
  \operatorname{Par}(s)}P\_B(s'\mid s)=1$.
- $o\in\mathcal{O}$: A (learned) parameter configuration for a GFN.
- $\Pi(o)\in\Delta(\mathcal{T})$: Probability measure over trajectories.
- $\pi_o$: The training distribution.
- $\mathcal{H}$: Maps a Markovian flow $F$ to its parameterization $o$.
- $(\mathcal{O}, \Pi, \mathcal{H})$: A flow paramterization for pointed DAG
  $G=(\mathcal{S}, \mathbb{A})$.
- $(G,R,\mathcal{O}, \Pi, \mathcal{H})$: A GFlowNet specification.
- $\mathcal{E}:\mathcal{S}\to \mathbb{R}$: An energy function mapping states to
  real values.
- $\mathcal{F}(s)$: free energy such that
  $e^{-\mathcal{F}(s)}:=\sum_{s^{\prime}: s^{\prime} \geq
  s}e^{-\mathcal{E}\left(s^{\prime}\right)}$
- $\mathcal{X}$: A set of conditioning variables.
- $G_x=(\mathcal{S}\_x,\mathcal{A}_x)$: A DAG indexed by $x\in\mathcal{X}$.
- $R_x(s)=R(s\mid x)$: A conditional reward function.

## Introduction

- GFNs sample a composite object $s$ such that $P_T(s)\propto R(s)$
- GFNs trade training complexity for sampling complexity, i.e. for MCMC
  methods, sampling can be expensive and the mixing time can be very high; if
  the modes share common structure, GFNs can learn that and sample them more
  efficiently
- GFNs (1) can be trained in an offline manner from one different from the GFN
  or target distribution, provided it has sufficient support and (2) they match
  the reward function in probability rather than finding a policy that
  maximizes reward.
- GFNs and TD methods rely on local coherence, e.g. detailed balance or flow
  matching conditions, between components and a training
  objective that estimates a global quantity of interest when those components
  cohere.

## Flow Networks and Markovian Flows

### Trajectories and Flows

- **Lemma 5** (proof on p.6): The sum of the forward probabilities of all trajectories
  starting at a state and ending in the sink is 1. Similarly, the sum of the
  backward probabilities of all trajectories starting at a state and ending in
  the source is 1.

$$
\begin{aligned}
\forall \tau=\left(s_1, \ldots, s_n\right) \in \mathcal{T}^{\text {partial }}
& \hat{P}_F(\tau) := \prod_{t=1}^{n-1} \hat{P}_F\left(s_{t+1} \mid s_t\right) \\
\forall \tau=\left(s_1, \ldots, s_n\right) \in \mathcal{T}^{\text {partial }}
& \hat{P}_B(\tau) := \prod_{t=1}^{n-1} \hat{P}_B\left(s_t \mid s_{t+1}\right) \\
\forall s \in \mathcal{S} \backslash\left\{s_f\right\} \quad \sum_{\tau \in
\mathcal{T}_{s, f}}
& \hat{P}_F(\tau)=1 \\
\forall s^{\prime} \in \mathcal{S} \backslash\left\{s_0\right\} \quad
\sum_{\tau \in \mathcal{T}_{0, s^{\prime}}}
& \hat{P}_B(\tau)=1
\end{aligned}
$$

- For every subset $A\subseteq\mathcal{T}$: $F(A)=\sum_{\tau\in A}F(\tau)$
- **Proposition 8**: Given a flow network $(G, F)$, the flow through a state is
  equal to both the total flow into the state and the total flow out of the
  state.

$$
\begin{aligned}
&\forall s \in \mathcal{S} \backslash\left\{s_f\right\} \quad
F(s)=\sum_{s^{\prime} \in \operatorname{Child}(s)} F\left(s \to
s^{\prime}\right) \\
&\forall s^{\prime} \in \mathcal{S} \backslash\left\{s_0\right\} \quad
F\left(s^{\prime}\right)=\sum_{s \in \operatorname{Par}\left(s^{\prime}\right)}
F\left(s \to s^{\prime}\right)
\end{aligned}
$$

### Flow Induced Probability Measures

- **Proposition 10**: $F(s_0)=F(s_f)=\sum_{\tau\in\mathcal{T}}=Z$. The
  normalizing constant $Z$ can turn the measure space $(\mathcal{T},
  2^\mathcal{T},F)$ into the probability space $(\mathcal{T},2^\mathcal{T},P)$.

### Markovian Flows

- Normally, defining a flow requires defining $\lvert\mathcal{T}\rvert$
  non-negative flows, but Markovian flows can be reduced by factorizing common
  paths according to $G$.
- A flow is Markovian if $P(s\to s'\mid\tau)=P(s\to s'\mid
  s)=P_F(s'\mid s)$ for any $s\neq s_0$, outgoing edge $s\to s'$, and
  trajectory $\tau=(s_0,s_1,\ldots,s_n=s)\in\mathcal{T}^{partial}$.
- **Proposition 16** (proofs on p.10-13): If a flow factorizes forward or backward and each
  transition only depends on the current state, the flow is Markovian.
  Conversely, if the flow is Markovian, it factorizes forwards and backwards
  and each transition depends on only the current state. More formally, the
  following statements are equivalent:

  1. $F$ is a Markovian flow.
  1. There exists a unique probability function $\hat{P}\_F$ consistent with
     $G$ such that for all complete trajectories
     $\tau=(s_0,\ldots,s_{n+1}=sf)$:

     $$
     P(\tau)=\prod_{t=1}^{n+1}\hat{P}_F(s_t\mid s_{t-1})
     $$

     Where $\hat{P}_F=P_F$.

  1. There exists a unique probability function $\hat{P}\_B$ consistent with
     $G$ such that for all complete trajectories
     $\tau=(s_0,\ldots,s_{n+1}=sf)$:

     $$
     P(\tau)=\prod_{t=1}^{n+1}\hat{P}_B(s_{t-1}\mid s_t)
     $$

     Where $\hat{P}_B=P_B$.

- **Corollary 17** (proof p.14): In a Markovian flow network, $(G, F)$,
  $P_T(s_n)=P_F(s_1\mid s_0)\cdots P_F(s_n\mid s_{n-1})$.
- **Proposition 18** (proofs on p.14-15): Given a pointed DAG
  $G=(\mathcal{S},\mathbb{A})$, a Markovian flow on $G$ is _completely_ and
  _uniquely_ specified by one of the following:
  1. The combination of the total flow $\hat{Z}$ and the forward transition
     probabilities $\hat{P}_F(s'\mid s)$ for all edges $s\to
     s'\in\mathbb{A}$.
  1. The combination of the total flow $\hat{Z}$ and the backward transition
     probabilities $\hat{P}_B(s\mid s')$ for all edges $s\to
     s'\in\mathbb{A}$.
  1. The combination of the terminating flows $\hat{F}(s\to s_f)$ for
     all terminating edges $s\to s_f\in\mathbb{A}^f$ and the backwards
     transition probabilities $\hat{P}_B(s\mid s')$ for all non-terminating
     edges $s\to s'\in\mathbb{A}^{-f}$.

### Flow Matching Conditions

- **Proposition 19** (proofs on p.16-17): Given a pointed DAG $G=(\mathcal{S},\mathbb{A})$ and a non-negative function $\hat{F}$, which takes in put $s\in\mathcal{S}$ or a transition $s\to s'\in\mathbb{A}$, then $\hat{F}$ corresponds to a flow _if and only if_ the **flow matching conditions** are satisfied:

$$
\begin{aligned}
\forall s^{\prime} > s_0, \quad \hat{F}\left(s^{\prime}\right)
&=\sum_{s \in \operatorname{Par}\left(s^{\prime}\right)} \hat{F}\left(s
\to s^{\prime}\right) \\
\forall s^{\prime} < s_f, \quad \hat{F}\left(s^{\prime}\right)
&=\sum_{s^{\prime \prime} \in \operatorname{Child}\left(s^{\prime}\right)}
\hat{F}\left(s^{\prime} \to s^{\prime \prime}\right) \\
\text{Then, given:}& \\
\hat{Z}&=\hat{F}(s_0) \\
\hat{P}_F\left(s^{\prime} \mid s\right)
&\coloneqq\frac{\hat{F}\left(s \to s^{\prime}\right)}{\hat{F}(s)} \\
\hat{F}\text{ uniquely defines a flow $F$:}& \\
F(\tau)=\hat{Z} \prod_{t=1}^{n+1} \hat{P}_F\left(s_t \mid s_{t-1}\right)
&=\frac{\prod_{t=1}^{n+1} \hat{F}\left(s_{t-1} \to
s_t\right)}{\prod_{t=1}^n \hat{F}\left(s_t\right)}
\end{aligned}
$$

- $\hat{P}_F$ and $\hat{P}_B$ are **compatible** if there exists a flow
  function $\hat{F}:\mathbb{A}\to\mathbb{R}^+$ such that

$$
\hat{P}_F\left(s^{\prime}
\mid s\right)=\frac{\hat{F}\left(s \to
s^{\prime}\right)}{\sum_{s^{\prime} \in \operatorname{Child}(s)} \hat{F}\left(s
\to s^{\prime}\right)}, \quad \hat{P}_B\left(s \mid
s^{\prime}\right)=\frac{\hat{F}\left(s \to
s^{\prime}\right)}{\sum_{s^{\prime \prime} \in
\operatorname{Par}\left(s^{\prime}\right)} \hat{F}\left(s^{\prime \prime}
\to s^{\prime}\right)}
$$

- **Proposition 21**: $\hat{F}$, $\hat{P}\_B$, $\hat{P}\_F$ jointly correspond to
  a flow _if and only if_ **detailed balance** holds:

$$
\forall s \to s^{\prime} \in \mathbb{A} \quad \hat{F}(s)
\hat{P}_F\left(s^{\prime} \mid s\right)=\hat{F}\left(s^{\prime}\right)
\hat{P}_B\left(s \mid s^{\prime}\right)
$$

### Backwards Transitions can be Chosen Freely

- Backward transitions can be chosen to achieve certain goals or to simplify
  other calculations. Some examples:
  - Goal of simplicity: make all parents of a node have equal weight.
  - Goal of shortest paths: more weight to shortest paths.
  - Goal of learning $P_F$ or $F$ easier: let a learner discover $P_B$.

### Equivalence Between Flows

- Two flows,$F_1,F_2\in\mathcal{F}(G)$, are equivalent if

$$
\forall s\to s'\in\mathbb{A}\quad F_1(s\to s')=F_2(s\to s')
$$

- Example:

{% marginfigure 'figure-3' 'papers/figures/gflownet-foundations/figure-3.png' 'Figure 3' %}

$$
\begin{array}{ccccc}
\hline \tau & F_1(\tau) & F_2(\tau) & F_3(\tau) & F_4(\tau) \\
\hline s_0, s_2, s_f & 1 & 4 / 5 & 1 & 6 / 5 \\
s_0, s_1, s_2, s_f & 1 & 6 / 5 & 1 & 4 / 5 \\
s_0, s_2, s_3, s_f & 1 & 6 / 5 & 2 & 9 / 5 \\
s_0, s_1, s_2, s_3, s_f & 2 & 9 / 5 & 1 & 6 / 5 \\
\hline
\end{array}
$$

- Given the preceding table and Figure 3, flows $F_1$ and $F_2$ are
  equivalent. $F_3$ and $F_4$ are equivalent, but not equivalent to $F_1$
  and $F_2$. Equivalence can be tested by summing up all trajectories that
  contain a particular edge and comparing that value between flows for
  every edge.
- $F\_2$ and $F\_4$ are Markovian. $F\_1$ and $F\_3$ are not Markovian.
  Intuitively, a flow cannot be Markovian if the probability of an edge $s\to
  s'$ changes depending on the partial trajectory from $s_0\to s$, i.e. the
  flow "remembers" more than the previous state. For example, take edge $s_2\to
  s_3$ in $F_2$. There are two partial trajectories leading to this edge,
  $s_0\to s_2$ and $s_0\to s_1\to s_2$. The probability of transitioning
  through this edge for each partial trajectory must be equal to one another
  and to $P_F(s'\mid s)$.

  $$
  \begin{aligned}
  P(s_2\to s_3\mid s_0\to s_2)
  &= \frac{F(s_0\to s_2\to s_3)}{F(s_0\to s_2)} \\
  &= \frac{F(s_0\to s_2\to s_3\to s_f)}{F(s_0\to s_2\to s_3\to s_f)+F(s_0\to s_2\to s_f)} \\
  &= \frac{6/5}{6/5 + 4/5} \\
  &= \boxed{\frac{3}{5}} \\
  P(s_2\to s_3\mid s_0\to s_1\to s_2)
  &= \frac{F(s_0\to s_1\to s_2\to s_3)}{F(s_0\to s_1\to s_2)} \\
  &= \frac{F(s_0\to s_1\to s_2\to s_3\to s_f)}{F(s_0\to s_1\to s_2\to s_3\to s_f)
  + F(s_0\to s_1\to s_2\to s_f)} \\
  &= \frac{9/5}{9/5 + 6/5} \\
  &= \boxed{\frac{3}{5}} \\
  P_F(s_3\mid s_2)
  &= \frac{P(s_2\to s_3)}{P(s_2)} \\
  &= \frac{F(s_2\to s_3)}{F(s_2)} \\
  &= \frac{F(s_0\to s_1\to s_2\to s_3\to s_f) + F(s_0\to s_2\to s_3\to s_f)}
  {F(s_0\to s_2\to s_f)+ F(s_0\to s_1\to s_2\to s_f) + F(s_0\to s_2\to s_3\to s_f) + F(s_0\to s_1\to s_2\to s_3\to s_f)} \\
  &= \boxed{\frac{3}{5}} \\
  \end{aligned}
  $$

- $F_1$, $F_2$, $F_3$, and $F_4$ coincide on the terminating flows, i.e. at
  $s_2\to s_f$ and $s_3\to s_f$.

- **Proposition 23** (proof on p.20): If two flow functions
  $F_1,F_2\in\mathcal{F}\_{Markov}(G)$ for a pointed DAG $G$ are equivalent,
  then they are equal. Furthermore, for _any_ flow function
  $F'\in\mathcal{F}(G)$, there exists a unique Markovian flow function
  $F\in\mathcal{F}_{Markov}(G)$ such that $F$ and $F'$ are equivalent. There
  are two important consequences of this:
  1. **Efficiency**: You only need to focus on Markovian flows, which decreases
     the requirements from specifying $F(\tau)$ for all trajectories to $F(s\to
     s')$ for all edges. Generally, this is exponentially smaller than $\lvert
     T\rvert$.
  1. **Simplicity**: In order to approximate or learn a Markovian flow, you
     need only learn the edge flow function, which is a much smaller object
     than the actual flow function.

## GFlowNets: Learning a Flow

- The goal is to find functions such as $F(s)$ or $P(s\to s'\mid s)$ using
  estimators $\hat{F}(s)$ and $\hat{P}(s\to s'\mid s)$, which may not
  correspond to a proper flow.
- These learning machines are called **GFlowNets**.
- Given a reward function $R:\mathcal{S}^f\to \mathbb{R}^+$, GFNs attempt to estimate:

$$
\forall s\in \mathcal{S}\quad F(s\to s')=R(s)
$$

- Because of equivalences, without loss of generality, it's prudent to have
  GFNs approximate Markovian flows only:

$$
\mathcal{F}_{Markov}(G,R)=\\{F\in\mathcal{F}_{Markov}(G),\;\forall
s\in\mathcal{S}^f\quad F(s\to s^f)=R(s)\\}
$$

### GFlowNets as an Alternative to MCMC Sampling

- MCMC suffers from mode-mixing, and GFNs replace long MCMC chains with a
  single learned configuration.
- GFNs benefit where there is common structure shared between modes of a
  distribution, i.e. where traditional ML could generalize about the structure
  of rewards.

### GFlowNets and flow-matching losses

- Given a pointed DAG $G=(\mathcal{S},\mathbb{A})$ with $s_0$, $s_f$, and $R: \mathcal{S}^f\to \mathbb{R}^+$, we say that $(\mathcal{O},\Pi,\mathcal{H})$ is a **flow parameritization** of $(G,R)$ if:
  1. $\mathcal{O}$ is a non-empty set.
  1. $\Pi$ is a function mapping each object $o\in \mathcal{O}$ to an element
     $\Pi(o)\in\Delta(\mathcal{T})$, the set of probability distributions on
     $\mathcal{T}$.
  1. $\mathcal{H}$ is an injective functional from $\mathcal{F}_{Markov}(G,R)$ to $\mathcal{O}$.
  1. For any $F\in\mathcal{F}_{Markov}(G,R)$, $\Pi(\mathcal{H}(F))$ is the
     probability measure associated with the flow $F$.
- Each object $o\in \mathcal{O}$ implicitly defines a **terminating state probability** measure:

$$
\forall s\in \mathcal{S}^f\quad P_T(s)\coloneqq \sum_{\tau\in \mathcal{T}:s\to
s_f\in\tau}\Pi(o)(\tau)
$$

- Only some parameterizations, $o=\mathcal{H}(F)$ for
  $F\in\mathcal{F}_{Markov}(G,R)$, satisfy $P_T(s)\propto R(s)\quad\forall
  s\in\mathcal{S}^f$.
- GFNs provide a solution to the (generally intractable) problem of sampling
  from a target reward function $R$ or its associated **energy function**:
  $\mathcal{E}(s)\coloneqq -\log R(s)\;\forall s\in\mathcal{S}^f$.
- Searching for an object $o\in
  \mathcal{H}(\mathcal{F}\_{Markov}(G,R))\subseteq\mathcal{O}$ is often simpler
  then directly approximating flows $F\in\mathcal{F}\_{Markov}(G,R)$.

#### Edge-flow Parameterization: $(\mathcal{O}\_{edge},\Pi\_{edge},\mathcal{H}\_{edge})$

- $\mathcal{O}_{edge}=\mathcal{F}(\mathbb{A}^{-f},\mathbb{R}^+)$, the set of
  functions from $\mathbb{A}^{-f}$ to $\mathbb{R}^+$.
- $\mathcal{H}\_{edge}:\mathcal{F}\_{Markov}(G,R)\to\mathcal{O}\_{edge}$, a
  function that takes a flow and returns a parameterization. Namely,
  $\mathcal{H}_{edge}(F): (s\to s')\in \mathbb{A}^{-f}\mapsto F(s\to s')$.
- $\Pi_{edge}:\mathcal{O}_{edge}\to\Delta(\mathcal{T})$, a function that takes
  a parameterization and returns a distribution over trajectories, i.e.
- $\hat{F}\in\mathcal{O}_{edge}$

$$
\begin{aligned}
\Pi_{edge}(\hat{F})(\tau)&\propto\prod_{t=1}^n P_{\hat{F}}(s_t\mid
s_{t-1})\quad\forall\tau=(s_0,\ldots,s_n=s_f)\in\mathcal{T} \\
P_{\hat{F}}\left(s^{\prime} \mid s\right)&= \begin{cases}\frac{\hat{F}\left(s
\rightarrow s^{\prime}\right)}{\sum_{s^{\prime \prime} \neq s_f}
\hat{F}\left(s \rightarrow s^{\prime \prime}\right)+R(s)} & \text { if }
s^{\prime} \neq s_f \\ \frac{R(s)}{\sum_{s^{\prime \prime} \neq s_f}
\hat{F}\left(s \rightarrow s^{\prime \prime}\right)+R(s)} & \text { if }
s^{\prime}=s_f\end{cases}
\end{aligned}
$$

#### Forward Transition Probability Parameterization: $(\mathcal{O}\_{PF},\Pi\_{PF},\mathcal{H}\_{PF})$

- $\mathcal{O}_{PF}=\mathcal{O}_1\times \mathcal{O}_2$.
- $\mathcal{O}_1=\mathcal{F}(\mathcal{S}\setminus\\{s_f\\}, \mathbb{R}^+)$ is
  the set of functions from all non-sink states to $\mathbb{R}^+$.
- $\mathcal{O}_2$ is the set of forward probability functions $\hat{P}_F$
  consistent with $G$.
- $\mathcal{H}\_{PF}:\mathcal{F}\_{Markov}(G, R)\to\mathcal{O}_{PF}$.
- $\Pi_{PF}:\mathcal{O}\_{PF}\to\Delta(\mathcal{T})$.

$$
\begin{aligned}
\mathcal{H}_{PF}(F)=\left(s\in\mathcal{S}\setminus\{s_f\}\mapsto F(s),
(s\to s')\in\mathbb{A}\mapsto P_F(s'\mid s)\right) \\
\forall\tau=(s_0,\ldots,s_n=s_f)\in\mathcal{T}\quad\Pi_{PF}(\hat{F},\hat{P}_F)(\tau)\propto\prod_{t=1}^n \hat{P}_F(s_t\mid s_{t-1})
\end{aligned}
$$

#### Transition Probabilities Parameterization: $(\mathcal{O}\_{PFB},\Pi\_{PFB},\mathcal{H}\_{PFB})$

- $\mathcal{O}\_{PFB}=\mathcal{O}_1\times
  \mathcal{O}_2\times\mathcal{O}_3=\mathcal{O}\_{PF}\times\mathcal{O}_3$
- $\mathcal{O}_1=\mathcal{F}(\mathcal{S}\setminus\\{s_f\\}, \mathbb{R}^+)$ is
  the set of functions from all non-sink states to $\mathbb{R}^+$.
- $\mathcal{O}_2$ is the set of forward probability functions $\hat{P}_F$
  consistent with $G$.
- $\mathcal{O}_3$ is the set of backward probability functions $\hat{P}_B$
  consistent with $G$.
- $\mathcal{H}\_{PFB}:\mathcal{F}\_{Markov}(G, R)\to\mathcal{O}_{PFB}$.
- $\Pi_{PFB}:\mathcal{O}\_{PFB}\to\Delta(\mathcal{T})$.

$$
\begin{aligned}
\mathcal{H}_{PFB}(F)=\left(\mathcal{H}_{PF}(F)),
(s\to s')\in\mathbb{A}^{-f}\mapsto P_B(s\mid s')\right) \\
\forall\tau=(s_0,\ldots,s_n=s_f)\in\mathcal{T}\quad\Pi_{PFB}(\hat{F},\hat{P}_F,\hat{P}_B)(\tau)\propto\prod_{t=1}^n \hat{P}_F(s_t\mid s_{t-1})
\end{aligned}
$$

#### Trajectory Balance Parameterization: $(\mathcal{O}\_{TB},\Pi\_{TB},\mathcal{H}\_{TB})$

- $\mathcal{O}\_{TB}=\mathcal{O}_1\times\mathcal{O}_2\times\mathcal{O}_3$
- $\mathcal{O}_1=\mathbb{R}^+$ parameterizes the partition function $\hat{Z}$.
- $\mathcal{O}_2$ is the set of forward probability functions $\hat{P}_F$
  consistent with $G$.
- $\mathcal{O}_3$ is the set of backward probability functions $\hat{P}_B$
  consistent with $G$.
- $\mathcal{H}\_{TB}:\mathcal{F}\_{Markov}(G, R)\to\mathcal{O}_{TB}$.
- $\Pi_{TB}:\mathcal{O}\_{TB}\to\Delta(\mathcal{T})$.

#### GFlowNet: $(G,R,\mathcal{O},\Pi,\mathcal{H})$

- $G=(\mathcal{S},\mathbb{A})$ is a pointed DAG with initial state $s_0$ and
  sink state $s_f$.
- $R:\mathcal{S}^f\to \mathbb{R}^+$: a target reward function.
- $(\mathcal{O},\Pi,\mathcal{H})$ a flow parameterization of $(G,R)$.
- GFlowNet can refer to both a configuration $o\in\mathcal{O}$ and the full
  specification $(G,R,\mathcal{O},\Pi,\mathcal{H})$
- If $o\in\mathcal{H}(\mathcal{F}\_{Markov}(G,R))$, then $P_T(s)\propto R(s)$
- To find a useful parameterization, you need to define a loss function
  $\mathcal{L}$ on $\mathcal{O}$ that equals 0 when
  $o\in\mathcal{H}(\mathcal{F}\_{Markov}(G,R))$ .

#### Flow-matching Losses

- Let $(G,R,\mathcal{O},\Pi,\mathcal{H})$ be a GFlowNet
- A **flow-matching loss** is any function $\mathcal{L}:\mathcal{O}\to \mathbb{R}^+$ such that

$$
\forall o\in \mathcal{O}\quad \mathcal{L}(o)=0\Leftrightarrow
\exists F\in\mathcal{F}_{Markov}(G,R)\quad o=\mathcal{H}(F)
$$

- $\mathcal{L}$ is **edge-decomposable** if there exists a function
  $L:\mathcal{O}\times \mathbb{A}\to \mathbb{R}^+$ such that:

$$
\forall o \in \mathcal{O} \quad \mathcal{L}(o)=\sum_{s \rightarrow s^{\prime} \in \mathbb{A}} L\left(o, s \rightarrow s^{\prime}\right)
$$

- $\mathcal{L}$ is **state-decomposable**, if there exists a function
  $L:\mathcal{O}\times \mathcal{S}\to \mathbb{R}^+$ such that:

$$
\forall o \in \mathcal{O} \quad \mathcal{L}(o)=\sum_{s \in \mathcal{S}} L(o, s),
$$

- $\mathcal{L}$ is **trajectory-decomposable**, if there exists a function
  $L:\mathcal{O}\times \mathcal{T}\to \mathbb{R}^+$ such that:

$$
\forall o \in \mathcal{O} \quad \mathcal{L}(o)=\sum_{\tau \in \mathcal{T}} L(o, \tau)
$$

- The objective is $\min_{o\in\mathcal{O}}\mathcal{L}(o)$; and for
  edge-decomposable loss, this would be
  $\min_{o\in\mathcal{O}}\mathbb{E}_{(s\to s')\sim\pi_T}\left[L(o,s\to
  s')\right]$ where $\pi_T$ is any full support probability distribution on
  $\mathbb{A}$. {% marginnote 'q' "TODO(danj): why doesn't $\pi_T\propto R(s)$" %}

##### Edge-flow Paramterization, State-decomposable Loss

- Given $(\mathcal{O}\_{edge},\Pi_{edge}, \mathcal{H}\_{edge})$ and
  $L_{FM}:\mathcal{O}\times \mathcal{S}\to \mathbb{R}^+$ defined for each
  $\hat{F}\in\mathcal{O}\_{edge}$ and $s'\in\mathcal{S}$ where

$$
L_{F M}\left(\hat{F}, s^{\prime}\right)=\left\{\begin{array}{l} \left(\log
\left(\frac{\delta+\sum_{s \in \operatorname{Par}\left(s^{\prime}\right)}
\hat{F}\left(s \rightarrow
s^{\prime}\right)}{\delta+R\left(s^{\prime}\right)+\sum_{s^{\prime \prime} \in
\operatorname{Child}\left(s^{\prime}\right) \backslash\left\{s_f\right\}}
\hat{F}\left(s^{\prime} \rightarrow s^{\prime \prime}\right)}\right)\right)^2
\quad \text { if } s^{\prime} \neq s_f, \\ 0 \quad \text { otherwise }
\end{array}\right.
$$

- $\delta\ge 0$ is a hyper-parameter determining the sensitivity to small flows.
- $\mathcal{L}_{FM}$ maps each $\hat{F}\in\mathcal{O}\_{edge}$ to

$$
\mathcal{L}_{F M}(\hat{F})=\sum_{s \in \mathcal{S}} L_{F M}(\hat{F}, s)
$$

##### Transitions Parameterization, Edge-decomposable Loss (Detailed Balance)

- Given $(\mathcal{O}\_{PFB},\Pi_{PFB}, \mathcal{H}\_{PFB})$ and
  $L_{DB}:\mathcal{O}\_{PFB}\times \mathbb{A}\to \mathbb{R}^+$ defined for each
  $(\hat{F},\hat{P}_F,\hat{P}_B)\in\mathcal{O}\_{PFB}$ and $s\to s'\in \mathbb{A}$
  where

$$
L_{D B}\left(\hat{F}, \hat{P}_F, \hat{P}_B, s \rightarrow s^{\prime}\right)=
\begin{cases}\left(\log \left(\frac{\delta+\hat{F}(s) \hat{P}_F\left(s^{\prime}
\mid s\right)}{\delta+\hat{F}\left(s^{\prime}\right) \hat{P}_B\left(s \mid
s^{\prime}\right)}\right)\right)^2 & \text { if } s^{\prime} \neq s_f, \\
\left(\log \left(\frac{\delta+\hat{F}(s) \hat{P}_F\left(s^{\prime} \mid
s\right)}{\delta+R(s)}\right)\right)^2 & \text { otherwise },\end{cases}
$$

- Again, $\delta\ge 0$ is a hyper-parameter.
- $\mathcal{L}_{DB}$ maps each
  $(\hat{F},\hat{P}\_F,\hat{P}\_B))\in\mathcal{O}\_{PFB}$ to

$$
\left.\mathcal{L}_{D B}\left(\hat{F}, \hat{P}, \hat{P}_B\right)\right)=\sum_{s \rightarrow s^{\prime} \in \mathbb{A}} L_{D B}\left(\hat{F}, \hat{P}, \hat{P}_B, s \rightarrow s^{\prime}\right)
$$

- Because the reward function does not completely specify the flow, this loss
  can be used with the $(\mathcal{O}\_{PF},\Pi_{PF},\mathcal{H}_{PF})$
  parameterization, using any function $\hat{P}\_B\in\mathcal{O}_3$ as input.

##### Trajectory Balance Parameterization, Trajectory-decomposable Loss

- Given $(\mathcal{O}\_{TB},\Pi_{TB}, \mathcal{H}\_{TB})$ and
  $L_{TB}:\mathcal{O}\_{TB}\times \mathcal{T}\to \mathbb{R}^+$ defined for each
  $(\hat{Z},\hat{P}_F,\hat{P}_B)\in\mathcal{O}\_{PFB}$ and $\tau\in\mathcal{T}$ where

$$
\quad L_{T B}\left(\hat{Z}, \hat{P}_F, \hat{P}_B, \tau\right)=\left(\log
\frac{\hat{Z} \prod_{t=1}^{n+1} \hat{P}_F\left(s_t \mid
s_{t-1}\right)}{R\left(s_n\right) \prod_{t=1}^n \hat{P}_B\left(s_{t-1} \mid
s_t\right)}\right)^2
$$

- $\mathcal{L}\_{TB}$ maps each $(\hat{Z},\hat{P}_F,\hat{P}_B)\in\mathcal{O}\_{TB}$ to

$$
\mathcal{L}_{TB}\left(\hat{Z}, \hat{P}_F, \hat{P}_B\right)=\sum_{\tau \in
\mathcal{T}} L_{T B}\left(\hat{Z}, \hat{P}_F, \hat{P}_B, \tau\right)
$$

#### Training by Stochastic Gradient Descent

- Evaluating $\mathcal{L}(o)$ is generally intractable (same with it's
  minimization) since only a subset of edges can be visited in finite time.
- For this reason, GFlowNets typically use stochastic gradient descent.
- For edge-decomposable losses:

$$
\nabla_o L\left(o, s \rightarrow s^{\prime}\right), \quad s \rightarrow
s^{\prime} \sim \pi_o
$$

- For trajectory-decomposable losses:

$$
\nabla_o L(o, \tau), \quad \tau \sim \pi_o
$$

- Here, $\pi_o$ is the training distribution.

### Extensions

- Time stamps to allow cycles (p.27).
- Stochastic rewards (p.28).
- Offline Training (p.28).
  - $\pi_T$ should be adaptive and could even be another GFN with a different
    reward function.

### Exploiting Data as Known Terminating States

- How can you use $(s,R(s))$ pairs for training?
  - If parameterizing $P_B$, sample $\tau$'s from $s$ and use those
    trajectories to update the flows and forward transition probabilities. The
    problem with this is that the trajectories do not have full support (p.29).

## Conditional Flows and Free Energies

- GFNs can recover $Z=F(s_0)$, but what about for arbitrary $s$? Are there
  conditional marginalization partition values? In general, no, because
  siblings may contribute to the flow of states downstream from $s$:

{% fullwidth 'papers/figures/gflownet-foundations/figure-4.png' '' %}

- Given a pointed DAG $G=(\mathcal{S},\mathbb{A})$, the partial order denoted $\ge$, and a function $\mathcal{E}: \mathcal{S}\to \mathbb{R}$, called the **energy function**, we define the **free energy** $\mathcal{F}(s)$ of a state $s$ as:

$$
e^{-\mathcal{F}(s)}:=\sum_{s^{\prime}: s^{\prime} \geq s}
e^{-\mathcal{E}\left(s^{\prime}\right)}
$$

### Conditional flow networks

- **Conditional flow network**: A specification of the following:
  - $\mathcal{X}$: A set of conditioning variables.
  - $\\{G_x=(\mathcal{S}\_x,\mathcal{A}_x)\\}$ indexed by $x\in\mathcal{X}$.
    - For each DAG $G_x$, $\mathcal{T}\_x$ is the set of complete trajectories in
      $G_x$ where $\mathcal{T}=\bigcup_{x\in\mathcal{X}}\mathcal{T}\_x$.
    - $(s_0\mid x)\in\mathcal{S}_x$ and $(s_f\mid x)\in\mathcal{S}_x$.
  - $F:\mathcal{X}\times\mathcal{T}\to \mathbb{R}^+$ such that $F(x,\tau)=0$
    when $\tau\notin\mathcal{T}\_x$
    - $F_x$: The function mapping each $\tau\in\mathcal{T}_x$ to $F(x,\tau)$.
- These are the same as regular GFNs and carry all the same properties, only
  now they are conditioned on $x$.

### Reward-conditional flow networks: $R(s\mid x)$

- This flow network uses the same graph, i.e. $G_x=G$ for every
  $x\in\mathcal{X}$ where $G=(\mathcal{S},\mathbb{A})$ is a pointed DAG, but
  there is a family $\mathcal{R}$ of reward functions over the terminal states
  conditional on $x\in\mathcal{X}$:

$$
\begin{aligned}
\mathcal{S}: \{R_x: \mathcal{S}^f&\to \mathbb{R}^+,x\in\mathcal{X}\} \\
\forall x\in\mathcal{X}\quad\forall s\in\mathcal{S}^f\quad &F_x(s\to s_f)=R_x(s)
\end{aligned}
$$

- As an example, set $R(s\mid \theta)=\exp(-\mathcal{E}_\theta(s))$ in the following energy-based model:

$$
P_\theta(s)=\frac{\exp(-\mathcal{E}_\theta(s))}{Z(\theta)}
$$

### State-conditional flow networks: $G_s$

- This flow network uses a subgraph $\\{G_s,s\in\mathcal{S}\\}$ anchored at $s$
  , i.e. $(s_0\mid s)=s$, and consisting of states $s'\ge s$ along with a
  conditional flow function $F:\mathcal{S}\times\mathcal{T}\to\mathbb{R}^+$
  where $\mathcal{T}=\bigcup_{s\in\mathcal{S}}\mathcal{T}_s$ and
  $\mathcal{T}_s$ is the set of complete trajectories in $G_s$ that satisfies
  $F_s(s'\to s_f)=F(s'\to s_f)$. Note that this means you cannot have a
  scenario where sibling contribute to total terminal flow.
- **Proposition 31** (proof on p.32): For any pointed DAG
  $G=(\mathcal{S},\mathbb{A})$ and flow $F$, we can define a state-conditional
  flow network.
