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
  - GCN layer with a skip connection: $$\mathbf{h}_v^{(l)}=\sigma\left(\underbrace{\sum_{u \in N(v)} \mathbf{W}^{(l)} \frac{\mathbf{h}_u^{(l-1)}}{|N(v)|}}_{F(x)}+\underbrace{\mathbf{h}_v^{(l-1)}}_{x}\right)$$
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
    - Could also augment with clustering coefficient, PageRank, Centrality, etc

# Lecture 6: GNN Augmentation and Training

-

# Lecture 7: Theory of Graph Neural Networks

# Lecture 8: Label Propagation on Graphs

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
