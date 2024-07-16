
Useful links:
-  [[IDDL 2017.pdf]]
-  [[Diploma thesis - moving forward]]

## IDDL in a nutshell

We are given a set of SPD matrices X of size\[dxd\].
In IDDL we want to:
- learn a dictionary B consiting of n atoms
- learn a pair of α, β parameters for ABLD metric
- encode the matrices into vectors where each dimension representing D<sup>(α,β)</sup> ( Χi || Bk )
- learn an objective function for the purpose of optimizing B atoms and their respective ABLD for classification

![[Drawing 2024-04-15 13.07.27.excalidraw]]

> [!NOTE]
> Instead of picking a specific metric for an application with IDDL an application specific metric is learnt.
> 

## General IDDL learning process

For optimization a Block Coordinate Descend is scheme is used, where its variable is updated independently by having all other values fixxed.

1) Read data (N samples of matrices X\[dxd\])

2) Create one-off matrix H (from data labels)

3) Initilize dictionary B of n atoms (n = m * num_of_classes) with m random samples per class
   
4) Initilize ABLD parameters ( α=1, β=1 recommened)
   
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
