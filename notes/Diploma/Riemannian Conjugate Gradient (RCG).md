
RCG method is guaranteed that new solutions have a lower objective along the geodesics of the Riemannian manifold.

#### Optimization problem

![[math-RCG generic.png]]

#### Updating Rule

![[math-RCG update.png]]

P(t): search direction
τΒ(t): TBSD++ -> Sd++

![[math-RCG direction.png]]

η(t): variable learning rate (learn by techniques such as Fletcher-Reeves)
grad*L*(B)Q: Riemannian gradient of the objective function
π(P(t-1), B(t-1), B(t)): parallel transport from Tx to Ty

![[math-RCG tools.png]]

> [!note]
> Computing the standard Euclidian gradient of the objective function L is the only requirement to perform RCG on Sd++.
