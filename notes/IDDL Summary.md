
## IDDL in a nutshell

We are given a set of SPD matrices X of size\[dxd\].
In IDDL we want to:
- learn a dictionary B consiting of n atoms
- learn a pair of α, β parameters for ABLD metric for each dictionary atom independently
- encode the matrices into vectors where each dimension representing D<sup>(α,β)</sup> ( Χi || Bk )
- learn an objective function for the purpose of optimizing B atoms and their respective ABLD for classification

![[Pasted image 20240716104156.png]]


> [!NOTE]
> Instead of picking a specific metric for an application with IDDL an application specific metric is learnt.
> 

## General IDDL learning process - Identification

For optimization a Block Coordinate Descend is scheme is used, where its variable is updated independently by having all other values fixxed.

1) Read data (N samples of matrices X\[dxd\])
2) Initialize dictionary B of n atoms (n = m * num_of_classes) with m random samples per class
3) Initialize ABLD parameters ( α=1, β=1 recommened)
4) Initialize any matrices needed for a loss function (e.g.: one-off matrix H from data labels)
5) Block Coordinate Descend for a specific amount of times
   - update dictionary B
   - update parameters α, β
   - update any other variables needed for calculations based on the objective function used

![[math-IDDL opt.png]]

## Reviewed cases

#### Ridge Regression Loss

- Notes: [[Ridge Regression Loss]]
- Code: [[Demo code notes]]

#### Structured SVM Loss

#### Information Divergence and Clustering

## Other related topics

[[ABLD (αβ-Log Determinant Divergence)]]
[[Riemannian Conjugate Gradient (RCG)]]
[[Linear Algebra Tools]]
[[Manifold Optimization Library]]
