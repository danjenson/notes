# GNNs

## Lecture 1: Introduction

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

## Lecture 2: Feature Engineering for ML in Graphs

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

### Link prediction

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

### Graph level features

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
    - O(|E|) running time
    - O(|V|) number of colors in memory
    - counting colors takes linear time with respect to |V|
    - total runtime is linear in |E|
    - far more computationally efficient than graphlet kernel

## Lecture 3: Node Embeddings

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

### How do you define similarity?

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
  - Objective: $\mathcal{L}=\sum_{u\in V}\sum_{v\in N_R(u)}-\log\left(\frac{\exp(\mathbf{z}_u^\intercal\mathbf{z}_v)}{\sum_{n\in V}\exp(\mathbf{z}_u^\intercal\mathbf{z}_n)}\right)$
    - $\max_f\sum_{u\in V}\log\Pr(N_R(u)\mid \mathbf{z}_u)$
    - $\mathcal{L}=\sum_{u\in V}\sum_{v\in N_R(u)}-\log\Pr(v\mid
    \mathbf{z}_u)$
    - $\Pr(v\mid \mathbf{z}+u)=\frac{\exp(\mathbf{z}_u^\intercal\mathbf{z}_v)}{\sum_{n\in
    V}\exp(\mathbf{z}_u^\intercal\mathbf{z}_n)}$
    - The denominator is expensive, so it can be approximated with noise
      contrastive estimation (NCE), i.e. instead of normalizing with respect to
      all nodes, just normalize against $k$ random negative samples
    - $\approx \log \left(\sigma\left(\mathbf{z}_u^{\mathrm{T}} \mathbf{z}_v\right)\right)-\sum_{i=1}^k \log \left(\sigma\left(\mathbf{z}_u^{\mathrm{T}} \mathbf{z}_{n_i}\right)\right), n_i \sim P_V$
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

### Embedding Entire Graphs

- Examples:
  - Classifying toxic vs. non-toxic molecules
  - Identifying anomalous graphs
- Approaches:
  - Run graph embedding technique and sum or average embeddings
  - Introduce a virtual node to represent the subgraph (linked to the nodes in
    that subgraph) and run graph embedding
    technique

### Matrix Factorization and Node Embeddings

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

### Limitations

- **Cannot obtain embeddings for nodes not in the training set**, i.e. shallow
  embeddings are only transductive
- **Cannot capture structural similarity**: neither DeepWalk nor node2vec
  capture structural similarity in node embeddings; this can be remedied by methods like
  struct2vec

## Lecture 4: Graph Neural Networks

## Lecture 5: A General Perspective on GNNs

## Lecture 6: GNN Augmentation and Training

## Lecture 7: Theory of Graph Neural Networks

## Lecture 8: Label Propagation on Graphs

## Lecture 9: Machine Learning with Heterogeneous Graphs

## Lecture 10: Knowledge Graph Embeddings

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
    - $\mathbf{r}_2 = \argmax_\mathbf{r} Re(<\mathbf{h},\mathbf{r},\overline{\mathbf{t}}>)$
    - $\mathbf{r}_1 = \argmax_\mathbf{r} Re(<\mathbf{t},\mathbf{r},\overline{\mathbf{h}}>)$
  - can model 1:N relations like DistMult
- **RotatE**: TransE in Complex space

- General rules:
  - Use TransE if the KG does not have many symmetric relationships

## Lecture 11: Knowledge Graphs

## Lecture 12: Fast Neural Subgraph Matching and Counting
