---
title: "CS 224W: Machine Learning with Graphs"
---

# Lecture 1: Introduction

- How do we take advantage of relational structure for better prediction?
- Modern deep learning is predicated on simple sequences and grids
- Map nodes to d-dimensional embeddings such that similar nodes in the network
  are embedded close together
- Common tasks:
  - node classification
  - link prediction
  - graph classification
  - clustering
  - graph generation
  - graph evolution
- **strongly connected**: a path from every node to ever other ndoe
- **weakly connected**: connected if we disregard edge directions

# Lecture 2: Feature Engineering for ML in Graphs

- Traditional features for ML in graphs with focus on undirected graphs
- Node level features:
  - **Goal**: characterize the structure and position of a node in the network
  - **importance-based features**:
    - node degree: counts neighboring nodes without capturing their importance
    - node centrality: takes node importance in a graph into account
      - eigenvector centrality: a node is important if it is us surrounded by
        important neighboring nodes; the largest eigenvalue is always positive and
        unique
      - betweenness centrality: a node is important if it lies on many shortest
        paths between other nodes
      - closeness centrality: a node is important if it has small shortest path
        lengths to all other nodes
  - **structure-based features**:
    - node degree
    - clustering coefficient: measures how connected neighboring nodes are
      (triangles)
    - graphlets: extends clustering coefficient to graph shapes beyond triangles;
      creates a graphlet degree vector (GDV); **graphlets** are rooted,
      connected, induced, non-isomorphic subgraphs
- **induced subgraph**: another graph formed from a subset of vertices and all
  the edges connecting the vertices in the subset
- **graph isomorphism**: two graphs which contain the same number of nodes
  connected in the same way are said to be isomorphic

## Link prediction

- For each pair of nodes, predict the score c(x,y) and sort
  by decreasing score, predict the top n pairs as new links, validate with true
  edges
- Tasks:
  - Links missing at random: remove a random set of links and then aim to
    predict them
  - Links over time: predict links that will manifest at the next time step
- Methods:
  - distance-based features
    - shortest-path distance between two nodes
  - local neighborhood overlap
    - common neighbors
    - Jaccard's coefficient
    - Adamic-Adar index
  - global neighborhood overlap
    - Katz index: count the number of walks of all lengths between a given pair
      of nodes; based on powers of adjacency matrix; if you used a discount
      factor, there is a closed form solution

## Graph level features

- **Goal**: We want features that characterize the structure of an entire graph.
- Kernel methods are widely-used for traditional ML for graph-level prediction
  - Design kernels instead of feature vectors?
- Kernel K(G, G') in R measures the similarity between graphs
  - **Goal**: design a graph feature vector $\phi(G)$
  - Kernel matrix **K**=(K(G, G')) must always be positive semi-definite
  - There exists a feature representation such that
    $K(G,G')=\phi(G)^\intercal\phi(G')$
  - Once the kernel is defined, off-the-shelf ML models such as kernel SVM can
    be used to make predictions
  - Examples:
    - Graphlet kernel
    - Weisfeiler-Lehman kernel
    - Random-walk kernel
    - Shortest-path graph kernel
  - Key idea: **bag-of-words** for a graph
    - BoW simply uses the word counts as feature for documents, without regard
      for order
    - A naive extension to graph: regard nodes as words
    - Bag of...
      - node colors (features)
      - node degrees
      - graphlet counts (**graphlet-kernel**); if a graph's node degree is
        bounded by d, then there exists an $O(nd^{k-1})$ algorithm to count all
        graphlets of size k
- Subgraph isomorphism is NP-hard
- **Goal**: can we design an efficient graph feature descriptor $\phi(G)$?
  - Can we generalize bag-of-node-degrees? Yes, this is called **color
    refinement**
  - Color refinement summarizes the structure of the K-hop neighborhood (uses a
    hash function for message passing aggregation)
  - After color refinement, Weisfeiler-Lehman (WL) kernel counts number of nodes with
    a given color.
  - WL kernel is the inner product of the color refinement count vectors
    - $O(\lvert E\rvert)$ running time
    - $O(\lvert V\rvert)$ number of colors in memory
    - counting colors takes linear time with respect to $\lvert V\rvert$
    - total runtime is linear in $\lvert E\rvert$
    - far more computationally efficient than graphlet kernel

# Lecture 3: Node Embeddings

- Graph representation learning eliminates the need to do feature engineering
- Want to encode nodes so they have certain properties (proximities) in the
  embedding space
- You train these embeddings with some measure of similarity, and a simple
  decoder would simply be the dot product of these embeddings, i.e. cosine
  similarity in the embedding space
- **Shallow embeddings** just lookup a node ID in a matrix and return the
  corresponding embedding, i.e. the encoder is just an embedding lookup
  - DeepWalk and node2vec are instances of this
- The following generates **task independent** embeddings that represent only
  their network structure; in particular, they are not using node labels for
  features

## How do you define similarity?

- Random walks: initiate random walks from all nodes in the graph and then
  $\mathbf{z}_u^\intercal \mathbf{z}_v$ represents the likelihood that nodes u
  and v co-occur on a random walk over the graph
  - Steps:
    1. Estimate the probability of visiting node v on a random walk starting from
       node u using some random walk strategy
    2. Optimize embeddings to encode these random walk statistics
  - Benefits:
    - Expressive: incorporates both local and higher order neighborhood information
    - Efficient: do not need to consider all node pairs when training, only node
      pairs that have co-occurred on random walks
  - Objective: $$\mathcal{L}=\sum_{u\in V}\sum_{v\in N_R(u)}-\log\left(\frac{\exp(\mathbf{z}_u^\intercal\mathbf{z}_v)}{\sum_{n\in V}\exp(\mathbf{z}_u^\intercal\mathbf{z}_n)}\right)$$
    - $\max_f\sum_{u\in V}\log\Pr(N_R(u)\mid \mathbf{z}_u)$
    - $\mathcal{L}=\sum_{u\in V}\sum_{v\in N_R(u)}-\log\Pr(v\mid
    \mathbf{z}_u)$
    - $$
      \Pr(v\mid \mathbf{z}+u)=\frac{\exp(\mathbf{z}_u^\intercal\mathbf{z}_v)}{\sum_{n\in
      V}\exp(\mathbf{z}_u^\intercal\mathbf{z}_n)}
      $$
    - The denominator is expensive, so it can be approximated with noise
      contrastive estimation (NCE), i.e. instead of normalizing with respect to
      all nodes, just normalize against $k$ random negative samples
    - $$\approx \log \left(\sigma\left(\mathbf{z}_u^{\mathrm{T}} \mathbf{z}_v\right)\right)-\sum_{i=1}^k \log \left(\sigma\left(\mathbf{z}_u^{\mathrm{T}} \mathbf{z}_{n_i}\right)\right), n_i \sim P_V$$
      - k is often chosen to be 5-20 and technically nodes on the random walk
        shouldn't be chosen
  - Update with (stochastic) gradient descent: $\mathbf{z}_u\leftarrow
    \mathbf{z}_u-\eta\pdv{L}{\mathbf{z}\_u}$
- DeepWalk: run fixed-length, unbiased random walks starting from each node
  - Problem: this notion of similarity is too constrained
- Node2Vec:
  - Goal: Embed nodes with similar network neighborhoods close in the feature
    space. Frame this as a maximum likelihood optimization problem.
  - Key observation: flexible notion of network neighborhood $N_R(u)$ of node u
    leads to rich node embeddings
  - Develop biased 2nd order random walk R to generate network neighborhood
    $N_R(u)$ of node u
  - Idea: use flexible, biased random walks that can trade off between local
    and global views of the network, i.e. **use a mix of DFS (macro-view) and BFS (micro-view)**
    - Instrument this with parameters p and q; p determines probability of
      returning to previous node and q is the ratio of BFS to DFS, i.e. moving
      outwards vs. inwards
    - This requires remembering where the walk came from (just the last step)
    - Lecture 3, slide 45 has an example of how p and q are used
  - Steps:
    1. Compute random walk probabilities
    2. Simulate $r$ random walks of length $l$ starting from each node $u$
    3. Optimize the node2vec objective using SGD
  - Benefits:
    - Linear-time complexity
    - All 3 steps are individually parallelizable
- Other random walks (links to papers on Lecture 3, slide 47):
  - based on node attributes
  - based on learned weights
  - based on 1-hop and 2-hop random walk probabilities
  - random walks on modified version of original network, i.e. struct2vec, HARP

## Embedding Entire Graphs

- Examples:
  - Classifying toxic vs. non-toxic molecules
  - Identifying anomalous graphs
- Approaches:
  - Run graph embedding technique and sum or average embeddings
  - Introduce a virtual node to represent the subgraph (linked to the nodes in
    that subgraph) and run graph embedding
    technique

## Matrix Factorization and Node Embeddings

- **Inner product decoder with node similarity defined by edge connectivity is
  equivalent to matrix factorization of adjacency matrix $\mathbf{A}$**
- Objective: extract a factorization $\mathbf{A}=\mathbf{Z}^\intercal\mathbf{Z}$
  where $\mathbf{A}$ is the adjacency matrix and $\mathbf{Z}$ is the embedding
  matrix
  - Generally, we can only learn $\mathbf{Z}$ approximately
- DeepWalk is equivalent of the matrix factorization of the following
  (explanation on Lecture 3, Slide 61):
  $$\log \left(\operatorname{vol}(G)\left(\frac{1}{T} \sum_{r=1}^T\left(D^{-1} A\right)^r\right) D^{-1}\right)-\log b$$
- Node2vec can also be formulated as a more complex matrix factorization; paper
  links on Lecture 3, Slide 61
- How do you use node embeddings?
  - Clustering/community detection
  - Node classification
  - Link prediction based on $f(\mathbf{z}_i,\mathbf{z}_j)$, where $f$ can be
    concatenate, Hardamard, sum/average, or distance

## Limitations

- **Cannot obtain embeddings for nodes not in the training set**, i.e. shallow
  embeddings are only transductive
- **Cannot capture structural similarity**: neither DeepWalk nor node2vec
  capture structural similarity in node embeddings; this can be remedied by methods like
  struct2vec

# Lecture 4: Graph Neural Networks

- Limitations of shallow embeddings:
  - $O(\lvert V\rvert d)$ parameters are required
  - No parameters are shared between nodes
  - Every node has its own unique embedding
  - Inherently transductive, i.e. can't generate embeddings for nodes that are
    not seen during training
  - Does not incorporate node features
- GNNs are node encoders based on multiple layers of non-linear transformations
  based on graph structure; these deep encoders can be combined with node
  similarity functions
- GNNs can embed nodes, graphs, and subgraphs
- GNN tasks:
  - node classification
  - link prediction
  - community detection
  - network similarity
- Machine learning can be formulated as an optimization problem:
  $$\min_\theta\mathcal{L}(\mathbf{y},f(\mathbf{x}))$$ where $\theta$ could be
  our shallow embeddings $\mathbf{Z}$ and the loss could be L2 loss:
  $\mathcal{L}(\mathbf{y},f(\mathbf{x}))=\lVert y-f(x)\rVert_2$
  - other loss functions include L1, Huber loss, max-margin (hinge) loss, cross
    entropy, etc.
  - $f$ could be a linear layer, MLP, or other NN like a GNN
- When there are no node features, you can do a one hot encoding of a nodes
- Because graphs have no spatial or temporal assignment by default, we should
  constrain our efforts to methods that are permutation invariant, this means
  that two "order plans" should be the same for the same graphs with differently
  labeled nodes/edges
- If $f(\mathbf{A}_i, \mathbf{X}_i)=f(\mathbf{A}_j, \mathbf{X}_j)$ for any order
  plan $i$ and $j$, we say that $f$ is a permutation invariant function
- **Definition**: For any graph function $f:\mathbb{R}^{\lvert V\rvert\times
  m}\times\mathbb{R}^{\lvert V\rvert \times\lvert V\rvert}\to\mathbb{R}^d$, $f$
  is **permutation-invariant** if $f(\mathbf{A}, \mathbf{X})=f(\mathbf{P}\mathbf{A}\mathbf{P}^\intercal, \mathbf{P}\mathbf{X})$ for any
  permutation $\mathbf{P}$, i.e. the value is the same regardless of whether you
  permute the adjacency matrix and features.
- For node representation, we learn a function that maps ndoes of $G$ to a
  matrix $\mathbb{R}^{m\times d}$.
- If we learn a function $f$ that maps a graph $G=(\mathbf{A},\mathbf{X})$ to a
  matrix $\mathbb{R}^{m\times d}$ and the output vector of a node at the same
  position in the graph remains unchanged for any order plan, then $f$ is
  **permutation equivariant**
- **Definition**: for any node function $f:\mathbb{R}^{\lvert V\rvert\times
  m}\times \mathbb{R}^{\lvert V\rvert\times\lvert V\rvert}\to\mathbb{R}^{\lvert
  V\rvert\times m}$, $f$ is **permutation-equivariant** if
  $\mathbf{P}f(\mathbf{A},\mathbf{X})=f(\mathbf{PAP}^\intercal, \mathbf{PX})$
  for any permutation $\mathbf{P}$, i.e. when you shuffle the input the output
  is shuffled in the same fashion.
- **Examples**:
  - $f(\mathbf{A}, \mathbf{X})=\mathbf{1}^\intercal \mathbf{X}$ is
    permutation-invariant
    - $f(\mathbf{PAP}^\intercal, \mathbf{PX})=\mathbf{1}^\intercal
    \mathbf{PX}=\mathbf{1}^\intercal \mathbf{X}=f(\mathbf{A}, \mathbf{X})$
  - $f(\mathbf{A}, \mathbf{X})=\mathbf{X}$ is permutation-equivariant
    - $f(\mathbf{PAP}^\intercal, \mathbf{PX})=\mathbf{PX}=\mathbf{P}f(\mathbf{A}, \mathbf{X})$
  - $f(\mathbf{A}, \mathbf{X})=\mathbf{AX}$ is permutation-equivariant
    - $f(\mathbf{PAP}^\intercal, \mathbf{PX})=\mathbf{PAP}^\intercal\mathbf{PX}=\mathbf{PAX}=\mathbf{P}f(\mathbf{A}, \mathbf{X})$
- GNNs consist of multiple permutation equivariant and invariant functions,
  unlike most deep ML, i.e. MLPs
- **Idea**: a node's neighborhood defines a computation graph and the goal is to
  learn how to propagate information across the graph to compute node features
- **Key idea**: generate node embeddings based on local network neighborhoods,
  each network neighborhood defines a computation graph (imagine trees rooted at
  nodes, where the children are the neighbors of nodes)
- Basic approach: average neighbor messages and apply a NN
- Given **a node**, the GCN that computes its embedding is **permutation
  invariant**
- Considering **all nodes**, the GCN computation is permutation equivariant
- $\mathbf{h}_v^{(0)}=\mathbf{x}_v$
- $$\mathbf{h}_v^{(k+1)}=\sigma\left(\mathbf{W}_k \sum_{u \in \mathbf{N}(v)} \frac{\mathbf{h}_u^{(k)}}{|\mathbf{N}(v)|}+\mathbf{B}_k \mathbf{~h}_v^{(k)}\right), \forall k \in\{0 . . K-1\}$$
  - Train $\mathbf{W}_k$ and $\mathbf{B}_k$ using SGD
- $\mathbf{z}_v=\mathbf{h}_v^{(K)}$
- The entire update in matrix form: $$H^{(k+1)}=\sigma\left(\tilde{A} H^{(k)} W_k^{\mathrm{T}}+H^{(k)} B_k^{\mathrm{T}}\right)$$ where $\tilde{\mathbf{A}}=\mathbf{D}^{-1}\mathbf{A}$.
  - In practice, this implies that efficient sparse matrix multiplication can be
    used $\tilde{\mathbf{A}}$ is sparse.
  - **Not all GNNs can be expressed in matrix form when the aggregation function
    is complex.**

## Unsupervised training

- When you don't have labels, you can use the graph structure as supervision
- If you say that "similar" nodes should have similar embeddings, then
- $$
  \mathcal{L}=\sum_{z_u, z_v} \operatorname{CE}\left(y_{u, v}, \operatorname{DEC}\left(z_u, z_v\right)\right)
  $$
  - $y_{u,v}=1$ when node $u$ and $v$ are similar
  - $\operatorname{CE}$ is the cross entropy loss
  - $\operatorname{DEC}$ is a decoder, such as inner product
  - node similarity can be based on:
    - Random walks (node2vec, DeepWalk, struct2vec)
    - Matrix factorization
    - Node proximity in the graph

## Supervised Training

- Directly train for a supervised task like node classification, e.g. is a drug
  safe or toxic?
- $$
  \mathcal{L}=-\sum_{v \in V} y_v \log \left(\sigma\left(\mathrm{z}_v^{\mathrm{T}} \theta\right)\right)+\left(1-y_v\right) \log \left(1-\sigma\left(\mathrm{z}_v^{\mathrm{T}} \theta\right)\right)
  $$
  - $y_v$ are labels
  - $\theta$ are classification weights
  - $\mathbf{z}_v$ is a node embedding

## Model Design

1. Define a neighborhood aggregation function.
2. Define a loss function on the embeddings.
3. Train on a set of nodes, i.e. a batch of compute graphs.
4. Generate embeddings for nodes as needed (even those we never trained on!)

## Inductive Capability

- The model is capable of induction when the same aggregation parameters are
  shared for all nodes (GraphSAGE), an added benefit is that the number of
  parameters is sublinear in $\lvert V\rvert$ and we can generalize to unseen
  nodes.
- A example is when you train on a protein interaction graph from model organism
  A and generate embeddings on a newly collected data about organism B

## GNNs vs CNNs

- The key difference is that we can learn an different weight function for each
  pixel surrounding the target node
- GNN formulation: $$\mathbf{h}_v^{(l+1)}=\sigma\left(\underbrace{\mathbf{W}_l}_{\text{node agnostic}} \sum_{u \in \mathbf{N}(v)} \frac{\mathbf{h}_u^{(l)}}{\lvert\mathbf{N}(v)\rvert}+\mathbf{B}_l \mathbf{h}_v^{(l)}\right), \forall l \in\{0, \ldots, L-1\}$$
- CNN formulation: $$\mathbf{h}_v^{(l+1)}=\sigma\left(\sum_{u \in \mathbf{N}(v)} \underbrace{\mathbf{W}_l^u}_{\text{pixel specific}} \mathbf{~h}_u^{(l)}+\mathbf{B}_l \mathbf{~h}_v^{(l)}\right), \forall l \in\{0, \ldots, L-1\}$$
- A CNN can be seen as a special GNN with fixed neighbor size and ordering
  - The size of the filter is pre-defined for a CNN
  - The advantage of GNN is it postprocesses arbitrary graphs with different
    degrees for each node
- CNN is not permutation invariant/equivariant, i.e. switching the order of
  pixels will lead to different outputs

## Summary:

- Use multiple layers for embedding nodes, propagating the previous hidden state
  to the next layer
- Mean aggregation for a GCN can be expressed in matrix form
- GNN is a general architecture of which CNN is a special case

# Lecture 5: A General Perspective on GNNs

## General GNN Framework

1. Message
2. Aggregation
3. Layer connectivity
4. Graph augmentation
5. Learning objective

- GNN Layer: compresses a set of vectors into a single vector
  - Message: $\mathbf{m}_u^{(l)}=\mathrm{MSG}^{(l)}\left(\mathbf{h}_u^{(l-1)}\right), u \in\{N(v) \cup v\}$, where the message could be a simple linear layer: $\mathbf{m}_u^{(l)}=\mathbf{W}^{(l)} \mathbf{h}_u^{(l-1)}$
  - Aggregation: $$\mathbf{h}_v^{(l)}=\operatorname{AGG}^{(l)}\left(\left\{\mathbf{m}_u^{(l)}, u \in N(v)\right\}\right)$$, where the aggregation function could be any permutation invariant operator like sum, mean, min, or max
    - Note you must first aggregate neighbors then aggregate that representation
      with the original node representation
  - One issue is that information about the source (locally rooted) node could
    get lost if it doesn't depend on its own embedding, so you should include it
    when computing the message
    - Usually different weights will be applied to the neighbors' messages and
      the source weights message/state
    - You can combine these using either concatenation or summation: $$\mathbf{h}_v^{(l)}=\operatorname{AGG}\left(\operatorname{AGG}\left(\left\{\mathbf{m}_u^{(l)}, u \in N(v)\right\}\right), \mathbf{m}_v^{(l)}\right)$$
  - Add non-linearity (activation) expressiveness, i.e. sigmoid, ReLU, or
    Sigmoid
- **Graph Convolutional Networks (GCN)**: $$\mathbf{h}_v^{(l)}=\sigma\left(\underbrace{\sum_{u \in N(v)}}_\text{aggregation} \underbrace{\mathbf{W}^{(l)} \frac{\mathbf{h}_u^{(l-1)}}{\lvert N(v)\rvert}}_\text{message}\right)$$
  - Note this is normalized by node degree
  - GCN is also assumed to have self-edges
- **GraphSAGE**: $$\mathbf{h}_v^{(l)}=\sigma\left(\mathbf{W}^{(l)} \cdot \operatorname{CONCAT}\left(\mathbf{h}_v^{(l-1)}, \mathrm{AGG}\left(\left\{\mathbf{h}_u^{(l-1)}, \forall u \in N(v)\right\}\right)\right)\right)$$
  - Note that neighbors are typically sampled
  - 3 types of neighborhood aggregation:
    - **Mean**: $\operatorname{AGG}=\sum\_{u\in \lvert
      N(v)\rvert}\frac{\mathbf{h}\_u^{l-1}}{\lvert N(v)\rvert}$
    - **Pool**: $$\operatorname{AGG}=\operatorname{Mean/Max}\left(\left\{\operatorname{MLP}\left(\mathbf{h}_u^{(l-1)}\right), \forall u \in N(v)\right\}\right)$$
    - **LSTM**: $$\mathrm{AGG}=\operatorname{LSTM}\left(\left[\mathbf{h}_u^{(l-1)}, \forall u \in \pi(N(v))\right]\right) $$
  - L2 normalization is applied at every layer:$$\mathbf{h}_v^{(l)} \leftarrow \frac{\mathbf{h}_v^{(l)}}{\left\|\mathbf{h}_v^{(l)}\right\|_2} \forall v \in V \text { where }\|u\|_2=\sqrt{\sum_i u_i^2}$$
    - Without this, the embedding vectors would have different scales
- **Graph Attention Networks**: $$\mathbf{h}_v^{(l)}=\sigma\left(\sum_{u \in N(v)} \underbrace{\alpha_{v u}}_\text{attention weights} \mathbf{W}^{(l)} \mathbf{h}_u^{(l-1)}\right)$$
  - In GCN/GraphSage, $\alpha_{v u}=\frac{1}{\lvert N(v)\rvert}$ is the
    weighting factor of node $u$'s message to node $v$, which implies that all
    neighbors are equally important
  - The idea with attention is that only a small portion of input maters and the
    rest should not affect the decision/calculation
  - **Goal**: specify arbitrary importance to different neighbors of each node
    in the graph
  - **Idea**: Compute the embedding $\mathbf{h}_v^{(l)}$ of each node in the
    graph following an attention strategy
    1. Let $a_{vu}$ be computed as a byproduct of the attention mechanism:
    - $e_{vu}$ indicates the importance of $u$'s message to $v$: $$e_{v u}=a\left(\mathbf{W}^{(l)} \mathbf{h}_u^{(l-1)}, \mathbf{W}^{(l)} \boldsymbol{h}_v^{(l-1)}\right)$$
    - $$e_{A B}=a\left(\mathbf{W}^{(l)} \mathbf{h}_A^{(l-1)}, \mathbf{W}^{(l)} \mathbf{h}_B^{(l-1)}\right)$$
    2. Normalize $e_{uv}$ into the final attention weights $\mathbf{\alpha}_{vu}$
       using the softmax: $\alpha_{uv}=\frac{\exp\left(e_{vu}\right)}{\sum_{k\in N(v)}\exp\left(e_{vk}\right)}$
    3. Calculated the weighted sum of neighbors: $$\mathbf{h}_v^{(l)}=\sigma\left(\sum_{u \in N(v)} \alpha_{v u} \mathbf{W}^{(l)} \mathbf{h}_u^{(l-1)}\right)$$
  - This approach is agnostic to the form of attention mechanism, $a$ - Could use a simple single-layer neural network, i.e. concatenate hidden
    state for target and neighbor node, run it through a linear layer to produce
    scalar $e$: $$\begin{aligned}
    & e\_{A B}=a\left(\mathbf{W}^{(l)} \mathbf{h}\_A^{(l-1)}, \mathbf{W}^{(l)} \mathbf{h}\_B^{(l-1)}\right) \\
    & =\operatorname{Linear}\left(\operatorname{Concat}\left(\mathbf{W}^{(l)} \mathbf{h}\_A^{(l-1)}, \mathbf{W}^{(l)} \mathbf{h}\_B^{(l-1)}\right)\right)
    \end{aligned} $$
    - **Multi-head attention** stabilizes the learning process of the attention
      mechanism
      - Create multiple attention scores (each replica with different parameters):
        $$
        \begin{aligned}
        & \mathbf{h}_v^{(l)}[1]=\sigma\left(\sum_{u \in N(v)} \alpha_{v u}^1 \mathbf{W}^{(l)} \mathbf{h}_u^{(l-1)}\right) \\
        & \mathbf{h}_v^{(l)}[2]=\sigma\left(\sum_{u \in N(v)} \alpha_{v u}^2 \mathbf{W}^{(l)} \mathbf{h}_u^{(l-1)}\right) \\
        & \mathbf{h}_v^{(l)}[3]=\sigma\left(\sum_{u \in N(v)} \alpha_{v u}^3 \mathbf{W}^{(l)} \mathbf{h}_u^{(l-1)}\right) \\
        \end{aligned}
        $$
      - Aggregate the output by concatenation or summation: $\mathbf{h}_v^{(l)}=\mathrm{AGG}\left(\mathbf{h}_v^{(l)}[1], \mathbf{h}_v^{(l)}[2], \mathbf{h}_v^{(l)}[3]\right)$
  - Benefits:
    - Allows for implicitly specifying different importance to different
      neighbors
    - Computationally efficient: the computation can be parallelized across all
      edges, and the same with aggregation
    - Storage efficient: sparse matrix operations do not require more than
      $O(V+E)$ entries to be stored
    - **Fixed** number of parameters, irrespective of graph size
    - Localized: only attends over local network neighborhoods
    - Inductive: it is a shared edge-wise mechanism and does not depend on
      global graph structure

## GNN Layer in Practice

- IN practice the following classic GNN layers are a great starting point
  - linear
  - batch norm: stabilizes training
  - dropout: prevents overfitting
  - activation: more expressive
  - attention: control relative importances
  - aggregation
- **Batch Normalization**: feature-wise normalization using mean and standard
  deviation by batch
- **Dropout**: regularizes network to prevent overfitting; during training,
  randomly drop neurons with probability $p$, at testing time, multiply all
  outputs by $p$
  - **In GNNs, dropout is applied to the linear layer in the message function**: $$\mathbf{m}_u^{(l)}=\mathbf{W}^{(l)} \mathbf{h}_u^{(l-1)}$$
- **Non-linear Acviation**:
  - ReLU: $\max(x, 0)$
  - Sigmoid: $\sigma(x)=\frac{1}{1+e^{-x}}$
  - Parametric ReLU (PReLU): $\max(x, 0)+a\cdot\min(x, 0)$ where $a$ is a
    trainable parameter
    - Performs better than ReLU
- Summary: modern deep learning modules can be included in GNN layer for better
  performance

## Stacking GNN Layers

- The standard way: stack GNN layers sequentially
- The issue: **GNNs suffer from over-smoothing problem where all the node
  embeddings converge to the same value**
- **Receptive field**: the set of nodes that determine the embedding of a node
  of interest
  - In a $K$-layer GNN, each node has a receptive field of $K$-hop neighborhood
  - The number of shared neighbors increases when you increase $K$
  - When the receptive field of two nodes have high-overlap, they are likely to
    have highly similar embeddings
  - Many GNN layers -> increase in receptive fields of nodes -> embeddings
    become highly similar -> over-smoothing
  - Lessons: be cautions when adding GNN layers; adding more does not always
    help
    1. Analyze the necessary receptive field to solve your problem
    2. Set the number of GNN layers $L$ to be a bit more than the receptive
       field we like, but not much larger
  - Question: how do we enhance the expressive power of a GNN if the number of
    layers is small, i.e. how to make a shallow GNN more expressive?
    1. Increase the expressive power within each layer by making transformation
       and aggregation a deep neural network, i.e. multi-layer MLP
    2. Add layers that do not pass messages, i.e. pre and post-processing
       layers, which work very well in practice
    - **pre-processing layers**: important when encoding node features like
      text/image
    - **post-processing layers**: important when reasoning/transformation over
      node embeddings are needed, e.g. graph classification, knowledge graphs
  - Question: what if my problem still requires many GNN layers?
    - Add skip connections in GNNs
    - Node embeddings in earlier GNN layers can sometimes better differentiate
      nodes
    - Increase the impact of earlier layers in the final node embeddings by
      adding shortcuts in the GNN,i.e. $F(x) + x$ instead of just $F(x)$
- **Skip connections**: intuitively, these create a mixture of models
  - $N$ skip connections implies $2^N$ possible paths
  - Each path could have up to $N$ modules
  - We automatically get a mixture of shallow and deep GNNs
  - GCN layer with a skip connection: $$\mathbf{h}_v^{(l)}=\sigma\left(\underbrace{\sum_{u \in N(v)} \mathbf{W}^{(l)} \frac{\mathbf{h}_u^{(l-1)}}{\lvert N(v)\rvert}}_{F(x)}+\underbrace{\mathbf{h}_v^{(l-1)}}_{x}\right)$$
- Another option is to directly skip to the last layer where the final layer aggregates from all
  the node embeddings from previous layers

## Graph Manipulation in GNNs

- Graph feature augmentation
- Graph structure manipulation
- Reasons for breaking the equality between the raw input graph and the
  computational graph
  - Feature level:
    - Input lacks features -> feature augmentation
      - Standard approaches:
        - Assign constant values to nodes
        - Assign unique IDs (one hot encodings) to nodes
  - Structure level:
    - Graph is too sparse -> inefficient message passing
      - Add virtual nodes / edges
      - Connect 2-hop neighbors via virtual edges
        - Intuition: instead of just using $A$, use $A+A^2$
        - Works well on bipartite graphs, e.g. authors-papers
      - Connect all nodes to a virtual node, after which all nodes will be a
        2-hop distance from one another (greatly improves message passing)
    - Graph is too dense -> message passing is too costly
      - Sample neighbors when doing message passing
      - Reduces computational cost and works well
    - Graph is too large -> cannot fit the computational graph into a GPU
      - Sample subgraphs to compute embeddings
- It's unlikely that the input graph happens to be the optimal computation graph
  for embeddings
- Constant node features:
  - Expressive power: medium. All nodes are identical but GNN can lear from
    graph structure.
  - Inductive learning: High. Simply assign constant to new nodes and run GNN.
  - Computational cost: Low. Only a 1 dimensional feature.
  - Use cases: any graph, inductive settings.
- One-hot node features:
  - Expressive power: High. Each node has a unique ID, so node specific
    information can be stored.
  - Inductive learning: Low. Cannot generalize to new nodes - new nodes
    introduce new IDs and GNN does't know how to embed unseen IDs.
  - Use cases: small graphs, transductive settings.
- Certain structures are hard to learn by GNN
  - Example: cycle counts
    - Can a GNN learn the length of a cycle that $v_1$ resides in?
      Unfortunately, no
    - Regardless of whether $v_1$ is in a tree, square, or infinite length
      (line) cycle, the computation graph (edges to two neighbors) is the same
    - **Could augment nodes with cycle counts**
    - **Could also augment with degree distribution clustering coefficient, PageRank, Centrality, etc**

# Lecture 6: GNN Augmentation and Training

## GNN Prediction

- Add a prediction head to node embeddings output
  - Different tasks require different prediction heads
- Suppose we wan to make a $k$-way prediction
  - Classification: classify among $k$ categories
  - Regression: regression $k$ targets
- Node-level prediction: $$\hat{\mathbf{y}}_v=\operatorname{Head}_\text{node}(\mathbf{h}_v^{(L)})=\mathbf{W}^{(H)}\mathbf{h}_v^{(L)}$$
  - $\mathbf{W}^{(H)}\in \mathbb{R}^{k\times d}$: maps node embeddings from
    $\mathbf{h}_v^{(L)}\in \mathbb{R}^d$ to $\hat{\mathbf{y}}_v\in \mathbb{R}^k$
    so that we can compute loss
- Edge-level prediction: $$\widehat{\boldsymbol{y}}_{u v}=\operatorname{Head}_{\text {edge } e}\left(\mathbf{h}_u^{(L)}, \mathbf{h}_v^{(L)}\right)$$
  - Options for HEAD:
    1. Concatenation + Linear: $$\widehat{\boldsymbol{y}}_{u v}=\operatorname{Linear}\left(\operatorname{Concat}\left(\mathbf{h}_u^{(L)}, \mathbf{h}_v^{(L)}\right)\right)$$
    2. Dot product:
       - One-way: $\hat{\mathbf{y}}_{uv}=\left(\mathbf{h}_u^{(L)}\right)^\intercal \mathbf{h}_v^{(L)}$
       - $k$-way: similar to multi-head attention, we use different $\mathbf{W}^{(i)}$: $$\begin{gathered}
            \widehat{\boldsymbol{y}}_{u v}^{(1)}=\left(\mathbf{h}_u^{(L)}\right)^T \mathbf{W}^{(1)} \mathbf{h}_v^{(L)} \\
            \widehat{\boldsymbol{y}}_{u v}^{(k)}=\left(\mathbf{h}_u^{(L)}\right)^T \mathbf{W}^{(k)} \mathbf{h}_v^{(L)} \\
            \widehat{\boldsymbol{y}}_{u v}=\operatorname{Concat}\left(\widehat{\boldsymbol{y}}_{u v}^{(1)}, \ldots, \widehat{\boldsymbol{y}}_{u v}^{(k)}\right) \in \mathbb{R}^k
            \end{gathered}$$
- Graph-level prediction:

  - Make predictions using all the node embeddings in our graph: $$\widehat{\boldsymbol{y}}_G=\operatorname{Head}_{\text {graph }}\left(\left\{\mathbf{h}_v^{(L)} \in \mathbb{R}^d, \forall v \in G\right\}\right)$$
  - The HEAD for graph prediction is similar to the AGG operation in a GNN layer

    1. Global mean pooling: $$\widehat{\boldsymbol{y}}_G=\operatorname{Mean}\left(\left\{\mathbf{h}_v^{(L)} \in \mathbb{R}^d, \forall v \in G\right\}\right)$$
    2. Global max pooling: $$\widehat{\boldsymbol{y}}_G=\operatorname{Max}\left(\left\{\mathbf{h}_v^{(L)} \in \mathbb{R}^d, \forall v \in G\right\}\right)$$

    3. Global sum pooling: $$\widehat{\boldsymbol{y}}_G=\operatorname{Max}\left(\left\{\mathbf{h}_v^{(L)} \in \mathbb{R}^d, \forall v \in G\right\}\right)$$

  - Issues:
    - Global pooling over large graphs will lose information
    - Example pathology with sum aggregation:
      - $$G_1=\{-1, -2, 0, 1, 2\}\implies \hat{y}_G=\operatorname{Sum}(\{-1,-2,0,1,2\})=0$$
      - $$G_2=\{-10, -20, 0, 10, 20\}\implies \hat{y}_G=\operatorname{Sum}(\{-1,-2,0,1,2\})=0$$
      - Cannot differentiate $G_1$ and $G_2$
  - Solution:
    - Aggregate all node embeddings hierarchically
    - For instance, above, if you partition the graphs into the first two and
      last three values and apply sum and ReLU to that, you get 3 for the first
      graph and 30 for the second (Lecture 6, Slide 35)

- **DiffPool**: hierarchically pool node embeddings
  - GNN A: computes node embeddings
  - GNN B: computes the cluster that a node belongs to (based on current layer
    embeddings)
  - These GNNs can be executed in parallel
  - Use clustering assignments from GNN B to aggregate node embeddings generated
    by GNN A
  - Create a single new node for each cluster, maintaining edges between
    clusters to generate a new pooled network
  - Jointly train GNN A and GNN B
- Supervised learning: external labels
- Unsupervised or self-supervised learning: labels come from graph itself, i.e.
  links
- Sometimes supervision is still used in unsupervised learning, e.g. train a GNN
  to predict node clustering coefficients
- **Examples of supervision signals**:
  - Node labels: which subject a citation belongs to
  - Edge labels: whether an edge is fraudulent
  - Graph labels: among molecular graphs, the drug likeness of graphs
- **Examples of unsupervised signals**:
  - Node level: node statistics such as clustering coefficient, PageRank, etc
  - Edge level: link prediction, i.e. hide edges and predict if it should be
    there
  - Graph level: graph statistics like whether two graphs are isomorphic
- **Advice**: Reduce your task to node/edge/graph labels, since they are easy to
  work with; e.g. we know some nodes form a cluster, we can treat the cluster
  that a node belongs to as a node label
- How to compute loss?
  - Classification loss, e.g. cross-entropy loss
  - Regression loss, e.g. MSE
- How to evaluate or measure success?
  - Accuracy
  - ROC AUC
  - Root mean squared error (RMSE)
  - Mean absolute error (MAE)
- Evaluating classification tasks:
  - Multi-class classification: accuracy $$\frac{1\left[\operatorname{argmax}\left(\widehat{\boldsymbol{y}}^{(i)}\right)=\boldsymbol{y}^{(i)}\right]}{N}$$
  - Binary classification:
    - Accuracy: (TP + TN) / (TP + TN + FP + FN) = (TP + TN) /
      $\lvert\text{Dataset}\rvert$
    - Precision (P): TP / (TP + FP)
    - Recall (R): TP / (TP + FN)
    - F1-Score: 2P \* R / (P + R)
    - Metric agnostic classification threshold: ROC AUC, which captures the
      TPR/FPR tradeoff
      - Intuition: The probability that a classifier will rank a randomly
        chosen positive instance higher than a randomly chosen negative one

## Dataset Splitting

- Fixed split:
  - Training used for optimizing GNN parameters
  - Validation used to evaluate model/tune hyper-parameters
  - Test used to report final performance
    - Sometimes, we cannot guarantee that the test set will really be held out
- Random split:
  - Split $k$ times randomly into training, validation, test and report average
    performance using random seeds for each
- **Problem**: nodes/edges in a graph are not independent, unlike sentences,
  images, etc
- **Solutions**:
  - **Transductive setting**:
    The input graph can be observed in all dataset splits, but we split on node
    labels
    - At training time, compute embeddings using the entire graph and train
      using the training set's node labels
    - At validation time, we compute embeddings using the entire graph and
      evaluate on a different subset of node labels
    - Only applicable to node/edge prediction tasks
  - **Inductive setting**: We create multiple graphs by breaking edges
    - Now we have 3 independent graphs
    - At training time, we compute embeddings using the graph over the
      training graph, using only those labels
    - At validation time, we compute embeddings using only the validation
      graph and evaluate performance on those labels
    - Applicable to node, edge, and graph tasks; works for graph tests because
      we have to test on unseen graphs
- Example: Link Prediction
  - Setting up link prediction requires hiding some edges and letting the GNN
    predict if the edges exist
  - Technique:
    1. Assign 2 types of edges in the original graph:
    - Message edges: used for GNN message passing
    - Supervision edges: used for computing objectives and not fed into GNN
    - After this step, only message edges will remain in the graph
    2. Split edges into train, validation, and test
    - Option 1: inductive link prediction split, i.e. 3 independent graphs, each
      of which will have supervision and message edges and the objective is to
      predict supervision edges with the respective subgraph
    - Option 2: transductive link prediction split (default setting), i.e.
      graph is visible in all splits, but you hold out various supervision
      edges for each data split
      1. At training time, use training message edges to predict training
         supervision edges
      2. At validation time, use training messages **and** training
         supervision edges to predict validation edges
      3. At test time, use training message edges, training supervision edges,
         and validation edges to predict test edges

# Lecture 7: Theory of Graph Neural Networks

## Understanding Expressiveness and Limitations

- How do we measure and maximize the expressive power of GNNs?
- **Key questions**:
  - How well can a GNN distinguish different graph structures?
  - How well can GNN node embeddings distinguish different node's local
    neighborhood structures?
- Fundamentally, a GNN generates node embeddings through a computational graph
  defined by the neighborhood (picture on Slide 19, Lecture 7)
  - Typical GNNs only see node features, not IDs
  - GNNs will generate the same embeddings for nodes whose computation graphs
    are identical even if the nodes are not
- Computational graphs are identical to rooted subtree structures around each
  node
- Most expressive GNNs map different rooted subtrees to different node
  embeddings (image on Slide 21, Lecture 7)
- **Key concept**: a function $f$ is injective if it maps different elements to
  different outputs, i.e. is 1:1; in this case, $f$ retains all the information
  about the input
- A maximally expressive GNN should map subtrees to node embeddings injectively
- **Key observation**: Subtrees of the same depth can be recursively
  characterized from the leaf nodes to the root nodes, e.g. (left: (2 neighbors),
  right: (3 neighbors)) and so on up the tree
  - If each step of the GNN's aggregation can fully retain the neighboring
    information, the generated node embeddings can distinguish different rooted
    subtrees, i.e. if each aggregation step of neighbors is injective, then the embedding
    process is injective
- **Key observation**: The expressive power of GNNs can be characterized by
  that of neighborhood aggregation functions and injective aggregation functions
  lead to the most expressive GNNs
- A neighbor aggregation can be abstracted as a function over a multi-set
- Analysis:
  - GCN uses element-wise mean-pooling, i.e. element-wise mean, linear layer,
    ReLU activation
    - Failure case is when it collapses distributions (example on Slide 32-34,
      Lecture 7)
  - GraphSAGE uses element-wise max-pooling
    - Failure case is when multi-sets contain the same base set of colors, i.e.
      it ignores the distribution of colors and collapses to just the minimal set
      of colors (image on Slide 36, Lecture 7)
  - **mean and max pooling are not injective** and hence any model based on them
    is not maximally expressive
- **Solution**: design a NN that can model injective multi-set functions
  - Any injective multi-set function can be expressed as
    $$\underbrace{\phi}_\text{non-linear function}\left(\underbrace{\sum_{x\in S}}_\text{sum over multi-set}\underbrace{f(x)}_\text{some non-linear function}\right)$$
  - Proof intuition: $f$ produces one-hot encodings of colors, and summation of
    one-hot encodings retains all the information about the input multi-set.
- **Universal approximation theorem**: 1-hidden-layer MLP with
  sufficiently-large hidden dimensionality and appropriate non-linearity
  $\sigma(\cdot)$ including ReLU and sigmoid can approximate any continuous
  function to an arbitrary accuracy.
  - We can use this to model the injective multi-set function:
    $\operatorname{MLP}_\phi\left(\sum_{x\in S}\operatorname{MLP}_f(x)\right)$
  - In practice, 100-500 hidden dimensions are sufficient

## Graph Isomorphism Network (GIN): The most expressive GNN

- Uses the results above to define the following injective aggregation function: $\operatorname{MLP}_\phi\left(\sum_{x\in S}\operatorname{MLP}_f(x)\right)$
- No failure cases!
- GIN is THE most expressive GNN in the class of message-passing GNNs we have
  introduced

## Relationship of Expressiveness to WL Graph Kernel

- tl;dr: GIN is a neural network version of the WL graph kernel
- **Color refinement algorithm in WL Kernel**:
  - Given a graph $G$ with a set of nodes $V$:
    - Assign an initial color $c^{(0)}(v)$ to each node $v$.
    - Iteratively refine node colors: $$c^{(k+1)}(v)=\operatorname{HASH}\left(c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u \in N(v)}\right)$$ where HASH is a perfect has injectively mapping different inputs to different colors
    - After $K$ steps of color refinement, $c^{(K)}(v)$ summarizes the structure
      of the $K$-hop neighborhood.
  - Process continues until a stable coloring is reached
  - Two graphs are considered isomorphic if they have the same set of colors
  - Illustration on Slides 46-48, Lecture 7
- **GIN uses a neural network to model the injective HASH function**;
  specifically, it models the injective function over the tuple $$(\underbrace{c^{(k)}(v)}_\text{root colors},
  \underbrace{\{c^{(k)}(u)\}_{u\in N(v)}}_\text{neighbor colors})$$
  - All together, the model is:
    $$
    \operatorname{GINConv}\left(c^{(k)}(v),\{c^{(k)}(u)\}_{u\in N(v)}\right)=\underbrace{\operatorname{MLP}_{\Phi}}_\text{provides one-hot input for next layer}\left((1+\epsilon) \cdot \operatorname{MLP}_f\left(c^{(k)}(v)\right)+\sum_{u \in N(v)} \operatorname{MLP}_f\left(c^{(k)}(u)\right)\right)
    $$
    - Here, $\epsilon$ is a learnable scalar
  - The full algorithm is, given a graph $G$ with a set of nodes $V$:
    - Assign an initial vector $c^{(0)}(v)$ to each node $v$.
    - Iteratively update node vectors with $$c^{(k+1)}(v)=\operatorname{GINConv}\left(\left\{c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u \in N(v)}\right\}\right)$$
      - Where GINConv maps different inputs to different embeddings, i.e. it is
        a differentiable color HASH function
    - After $K$ steps of GIN iterations, $c^{(K)}(v)$ summarizes the structure
      of the $K$-hop neighborhood.
- **GIN can be understood as a differentiable neural version of the WL graph
  kernel**:

|-----------------|-----------------------------------|-----------------|
| model | update target | update function |
|-----------------|-----------------------------------|-----------------|
| WL Graph Kernel | Node colors (one-hot) | HASH |
| GIN | Node embeddings (low-dim vectors) | GINConv |
|-----------------|-----------------------------------|-----------------|

- Advantages of GIN over WL graph kernel are:
  - Node embeddings are **low-dimensional** hence than can capture fine-grained
    similarity of different nodes.
  - Parameters of the update function can be learned for the downstream tasks.
- Because of the relationship between GIN and the WL graph kernel, their
  expressivity is exactly the same; namely, if two graphs can be distinguished
  by GIN, then they can be by the WL Kernel and vice versa
- How powerful is this? Why is this important?
  - WL kernel has been both theoretically and empirically shown to distinguish
    most real-world graphs
  - Hence, GIN is also power enough to distinguish most real-world graphs

## Looking forward

- Can the expressive power of GNNs be further improved? Some basic graph
  structures like difference in cycles cannot be distinguished by current GNNs
  (will address in Lecture 15)

## Summary

- GIN designs a NN that can model an injective multi-set function
- GIN is the most expressive GNN model
- The key is to use element-wise sum pooling instead of mean/max-pooling
- GIN is closely related to the WL kernel
- Both GIN and WL graph kernels can distinguish most real-world graphs

# Lecture 8: Label Propagation on Graphs

- **Question**: Given a network with labels on some nodes, how do we assign
  labels to all other nodes in the network, e.g. fraudsters in a social network?
- Node embeddings is one method to solve this problem; can we further use
  network topology?
- Given the labels of some nodes, let's predict the labels of unlabelled nodes
  - Transductive node classification (also called semi-supervised) node
    classification
- **Intuition**: correlations exist in networks, i.e. connected nodes tend to
  share the same label.
- 3 Techniques:
  - Label propagation
  - Correct & Smooth
  - Masked label prediction

# Lecture 9: Machine Learning with Heterogeneous Graphs

# Lecture 10: Knowledge Graph Embeddings

- Heterogeneous graphs are graphs with multiple relation types, each of which
  gets different network weights
- Nodes are labeled with types and edges capture relationships
- Examples:
  - Nodes: drug, disease, event, protein pathways
  - Relation types: has_func, causes, assoc, treats, is_a
- **Problem**: knowledge graphs are often incomplete and many true edges are
  missing; enumerating all relationships and/or facts may also be intractable
- **Task**: given an enormous KG, can we complete the KG, i.e. for a given
  (head, relation), can we predict the tail?
- **Key ideas**:
  - model entities and relations in an embedding/vector space
  - associate entities and relations with shallow embeddings
  - no GNN is learned here
- **Models**:

  | model    | score                                                                       | embedding                                                                                 | sym | antisym. | inv. | compos. | 1:N |
  | -------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --- | -------- | ---- | ------- | --- |
  | TransE   | $-\lVert\mathbf{h} + \mathbf{r}-\mathbf{t}\rVert$                           | $\mathbf{h},\mathbf{r},\mathbf{t}\in \mathbb{R}^k$                                        | x   | o        | o    | o       | x   |
  | TransR   | $-\lVert\mathbf{M}_r\mathbf{h} + \mathbf{r} - \mathbf{M}_r\mathbf{t}\rVert$ | $\mathbf{h},\mathbf{r},\mathbf{t}\in \mathbb{R}^k,\mathbf{M}_r\in \mathbb{R}^{d\times k}$ | o   | o        | o    | o       | o   |
  | DistMult | $<\mathbf{h},\mathbf{r},\mathbf{t}>$                                        | $\mathbf{h},\mathbf{r},\mathbf{t}\in \mathbb{R}^k$                                        | o   | x        | x    | x       | o   |
  | ComplEx  | $Re(<\mathbf{h},\mathbf{r},\overline{\mathbf{t}}>)$                         | $\mathbf{h},\mathbf{r},\mathbf{t}\in \mathbb{C}^k$                                        | o   | o        | o    | x       | o   |

- Relationships:

  - Symmetric:
    - $r(h,t)\implies r(t,h)\quad\forall h,t$
    - mother $\xrightarrow[\text{spouse}]{}$ father, father $\xrightarrow[\text{spouse}]{}$ mother
  - Antisymmetric:
    - $r(h,t)\implies \lnot r(t,h)\quad\forall h,t$
    - father $\xrightarrow[\text{child}]{}$ son, then **not** son $\xrightarrow[\text{child}]{}$ father
  - Inverse:
    - $r_2(h,t)\implies r_1(t,h)$
    - professor $\xrightarrow[\text{advisor}]{}$ student $\implies$ student $\xrightarrow[\text{advisee}]{}$ professor
  - Composable:
    - $r_1(x,y)\land r_2(y,z)\implies r_3(x,z)\quad\forall x,y,z$
    - father$\xrightarrow[\text{wife}]{}$mother$\xrightarrow[\text{mother}]{}$mother-in-law
  - 1:N:
    - $r(h,t_1),r(h,t_2),\ldots, r(h,t_n)$ are all true.
    - father $\xrightarrow[\text{child}]{}$ son **and** father $\xrightarrow[\text{chld}]{}$ daughter

- **TransE**: models translation of any relation in the **same** embedding space
  - cannot model symmetry, i.e. family, roommate, unless $h=t$ or $r=0$
  - cannot model 1:N relations, $t_1=h+r=t_2$ when $t_1\ne t_2$
  - can model antisymmetric: $h + r = t$, but $t + r \ne h$
  - can model inverse relationships by flipping sign of $r$
  - can model composition
- **TransR**: model entities as vectors in the entity space $\mathbb{R}^d$ and
  model each relation as a vector in relation space $\mathbf{r}\in \mathbb{R}^k$
  with $\mathbf{M}_r\in \mathbb{R}^{k\times d}$ as the projection matrix.
  - can model symmetric relations by projecting head and tail to same location
    in relation space (note that different symmetric relationships may have
    different $\mathbf{M}_r$)
  - can model antisymmetric relations the same as TransE, but in the relation space
  - can model 1:N by projecting all tails to the same location in the relation
    space
  - can model inverse relations the same as TransE, but in the relation space
  - can model composition relations; TransR models a triple with linear
    functions and they are chainable, proof on slide 37 of 10-kg.pdf
- **DistMult**:
  - TransE and TransR use negative of L1/L2 distance
  - Another strategy is to adopt **bilinear** modeling, making the score
    function a 3-way dot product
    - **Intuition**: can be viewed as a cosine similarity between $h\cdot r$ and
      $t$ where $h\cdot r$ is defined as $h_i\cdot r_i$
  - This defines half spaces, where if you are on the same side of the half
    space as $h\cdot r$ you are positive, otherwise negative
  - cannot model antisymmetric relations
  - cannot model inverse relations (i.e. advisor, advisee would be the same
    relation)
  - cannot model composition relations
    - **intuition**: DistMult defines a hyperplane for each (head, relation),
      and the union of the hyperplane induced by multi-hops of relations, e.g.
      ($r_1,r_2$) cannot be expressed using a single hyperplane, i.e. the union
      of hyperplanes cannot be captured as a hyperplane that captures the
      desired half-space
  - can model 1:N relations
  - can model symmetric relations
- **ComplEx**: models entities and relations in $\mathbb{C}^k$
  - score function is $f_r(h,t)=Re(\sum_i \mathbf{h}_i\cdot \mathbf{r}_i\cdot
    \overline{\mathbf{t}}_i)$
  - similar to DistMult, ComplEx cannot model compositions
  - can learn antisymmetric relations due to complex conjugate
    - high: $f_r(h,t)=f_r(h,t)=Re(\sum_i \mathbf{h}_i\cdot \mathbf{r}_i\cdot \overline{\mathbf{t}}_i)$
    - low: $f_r(h,t)=f_r(h,t)=Re(\sum_i \mathbf{t}_i\cdot \mathbf{r}_i\cdot \overline{\mathbf{h}}_i)$
  - can learn symmetric relations - when $Im(\mathbf{r})=0$:
    $$
    \begin{aligned}
    f_r(\mathbf{h}, \mathbf{t})
    &=\operatorname{Re}\left(\sum_i \mathbf{h}_i \cdot \mathbf{r}_i \cdot \overline{\mathbf{t}}_i\right) \\
    &=\sum_i \operatorname{Re}\left(\mathbf{r}_i \cdot \mathbf{h}_i \cdot \overline{\mathbf{t}}_i\right) \\
    &=\sum_i \mathbf{r}_i \cdot \operatorname{Re}\left(\mathbf{h}_i \cdot \overline{\mathbf{t}}_i\right) \\
    &=\sum_i \mathbf{r}_i \cdot \operatorname{Re}\left(\overline{\mathbf{h}}_i \cdot \mathbf{t}_i\right) \\
    &=\sum_i \operatorname{Re}\left(\mathbf{r}_i \cdot \overline{\mathbf{h}}_i\cdot \mathbf{t}_i\right)=f_r(t, h)
    \end{aligned}
    $$
  - can model inverse relations with $\mathbf{r}_1=\overline{\mathbf{r}}_2$
    - $$\mathbf{r}_2 = \arg\max_\mathbf{r} Re(<\mathbf{h},\mathbf{r},\overline{\mathbf{t}}>)$$
    - $$\mathbf{r}_1 = \arg\max_\mathbf{r} Re(<\mathbf{t},\mathbf{r},\overline{\mathbf{h}}>)$$
  - can model 1:N relations like DistMult
- **RotatE**: TransE in Complex space

- General rules:
  - Use TransE if the KG does not have many symmetric relationships

# Lecture 11: Knowledge Graphs

# Lecture 12: Fast Neural Subgraph Matching and Counting

# Lecture 13: GNNs for Recommender Systems

# Lecture 14: Deep Generative Models for Graphs

## Overview

- **Question**: How do we generate realistic graphs?
- Applications: drug discovery, material design, social network modeling
  - Insights, predictions, simulations, anomaly detection
- **Goal 1**: Realistic graph generation; generate graphs that are similar to a
  set of graphs
- **Goal 2**: Goal-directed graph generation; generate graphs that optimize
  given objectives/constraints
- $p_\text{data}(x)$ is the (unknown) data distribution
- $p_\text{model}(x;\theta)$ is the model, parameterized by $\theta$, that we
  use to approximate $p_\text{data}(x)$
- The objective is to make $p_\text{model}(x;\theta)$ as close to $p_\text{data}(x)$ as
  possible
- One possible way to do this is maximum likelihood: $$\boldsymbol{\theta}^*=\underset{\boldsymbol{\theta}}{\arg \max } \mathbb{E}_{x \sim p_{\text {data }}} \log p_{\text {model }}(\boldsymbol{x} \mid \boldsymbol{\theta})$$
- How do we sample from $p_\text{model}(x;\theta)$, i.e. a complex distribution?
  1. Sample form a simple noise distribution $\mathbf{z}_i\sim \operatorname{Normal}\left(0,1\right)$
  2. Transform the noise $\mathbf{z}_i$ via $f(\cdot)$ so $x$ follows the
     complex distribution
  3. Use Deep NN to train $f(\cdot)$
- **Auto-regressive models**: $p_\text{model}(x; theta)$ is used for both density estimation and sampling
  - Includes models like Variational Auto Encoders (VAEs) and Generative
    Adversarial Nets (GANs)
  - Idea: Chain rule: the joint distribution is a product of conditional
    distributions: $$p_{\text {model }}(\boldsymbol{x} ; \theta)=\prod_{t=1}^n p_{\text {model }}\left(x_t \mid x_1, \ldots, x_{t-1} ; \theta\right)$$
  - In our case $x_t$ will be the $t$-th action, i.e. add node or edge

## Graph RNN

- Two RNNs, one for generating nodes, the other for edges for each node,
  connecting to previous nodes in the graph, illustration on Slide 26, Lecture
  14
- At each node-level step, a node is added to the graph until the stop token is
  received by the RNN
- At each edge-level step, the edge RNN decides whether to connect the current
  node to a previously seen node
- Together, this can be seen as a sequence (nodes) of sequences (edges)
- This can also be imagined as creating the adjacency matrix
- Basic RNN cell:
  - $s_t = \sigma(W\cdot x_t + U\cdot s_{t-1})$
  - $y_t = V\cdot s_t$
  - LSTM and GRU are more advanced RNN cells
- How to use an RNN on graphs?
  - Let input be the previous steps output
  - Initialize the sequence with the start of sequence (SOS) token
  - Use end of sequence token (EOS) as an extra RNN output
    - If EOS=0, continue generation
    - If EOS=1, stop generation

### Edge-level RNN

- **Goal**: Model $$p_{\text {model }}(\boldsymbol{x} ; \theta)=\prod_{t=1}^n p_{\text {model }}\left(x_t \mid x_1, \ldots, x_{t-1} ; \theta\right)$$
- Let $y_t=p_\text{model}(x_t\mid x_1,\ldots,x_{t-1};\theta)$
- Then we need to sample $x_{t+1}$ from $y_t: x_{t+1}\sim y_t$
- **Each step outputs the probability of a single edge**
- Then we sample from that distribution and feed the sample to the next step;
  illustration on Slide 31, Lecture 14
- To train the model, use teacher forcing, i.e. correct the output $\hat{y}$
  with the true $y$ and correct the next step's input to be correct as well,
  i.e. if $y_t=0$, then $x_{t+1}=0$
- Use binary cross entropy loss to train RNN: $$L=-\left[y_1^* \log \left(y_1\right)+\left(1-y_1^*\right) \log \left(1-y_1\right)\right]$$
- If $y_1^*=1$, we minimize $-\log(y_1)$ by making $y_1$ larger
- If $y_1^*=-$, we minimize $-\log(1-y_1)$ by making $y_1$ smaller
- This fits the RNNs predictions to the true data (edges)

### Process

1. Add a new node: run node RNN for a step and use its output to initialize an
   edge RNN.
2. Add new edges for the new node: run edge RNN to predict if the new node will
   connect to each of the previous nodes.
3. Add a new node: we use the last hidden state of the edge RNN to run the node
   RNN for another step.
4. Stop graph generation: if the edge RNN outputs EOS at step 1, we know no
   edges are connected to the new node. We stop the graph generation.

- Illustrated on slides 35-42, Lecture 14
- At test time, replace input with GNN's own sampled predictions (Slide 43,
  Lecture 14)

### Tractability

- If any newly added node can connect to _any_ previous node, this quickly makes
  generation intractable; you need to generate a full adjacency matrix and the
  dependencies are long and complex
- **Solution**: use BFS node ordering, which reduces the number of possible node
  orderings from $O(n!)$ to the number of distinct BFS orderings
  - Only requires memory of last two steps instead of $n-1$ steps
  - In other words, you only consider connecting to nodes on the BFS frontier;
    the nodes that are not connected to nodes on the frontier are not considered

### Evaluation

- **Goal**: Define similarity metrics for graphs
- **Solution**:
  - Visual similarity
  - Graph statistics similarity
- GraphRNN is able to train grids, unlike Kronecker, MMSB, and B-A methods, and
  also does well on other graphs

### Applications: Drug Discovery

- **Question**: Can we learn a model that can generate valid and realistic
  molecules with optimized property scores?
- **Goal directed graph generation**:
  - Optimize a given objective (high scores), e.g. drug-likeness
  - Obey underlying rules (valid), e.g. chemical validity rules
  - Are learned from examples (realistic), e.g. imitating a molecule graph
    dataset
- The hard part: objectives like drug-likeness are governed by physical laws
  which are assumed to be unknown to us
- **Idea**: Reinforcement learning
  - An ML agent observes the environment, takes an action to interact with the
    environment, and receives positive or negative reward
  - The agent learns from this loop
  - **Key idea**: the agent can directly learn from the environment, which is a
    blackbox to the agent

### Solution: Graph Convolutional Policy Network (GCPN)

- GNN captures graph structural information
- RL guides generation toward the desired objectives
- Supervised training imitates examples in given datasets
- GCPN vs GraphRNN:
  - Both generate graphs sequentially
  - Both imitate a given graph dataset
  - GCPN uses GNN to predict the generation action
    - Pros: GNN is more expressive than RNN
    - Cons: GNN takes longer time to compute than RNN
  - GCPN further uses RL to direct graph generation to our goals, which enables
    goal-directed graph generation
  - Illustration on Slide 67, Lecture 14
- GCPN process:
  1. Insert nodes
  2. Use GNN to predict which nodes to connect
  3. Take an action (check chemical validity)
  4. Compute reward
- GCPN rewards = final reward + step rewards
  - At each step, assign a small positive reward for valid actions, which trains
    it to take valid actions
  - At the end, assign positive rewards for highly desired properties
- GCPN training (illustration on Slide 71, Lecture 14):
  1. Supervised training: train the policy by imitating the action given by
     real, observed graphs and use the gradient (similar to GNN)
  2. RL Training: train the policy to optimize rewards, using the standard
     policy gradient algorithm.
- Constrained optimization: edit a given molecule for a few steps to achieve
  higher property score

## Summary

- Complex graphs can be generated using sequential generation with deep RL
- Each step a decision is made based on hidden state, which can be
  - Implicit: vector representation, decode with RNN
  - Explicit: intermediate generated graphs, decode with GCN
- Possible tasks:
  - Imitating a set of given graphs
  - Optimizing graphs towards given goals

# Lecture 15: Advanced Topics in GNNs
