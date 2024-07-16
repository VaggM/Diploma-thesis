Many metrics have been proposed when comparing two SPD matrices in a Riemannian manifold.
The αβ-LogDet Divergence is a flexible metric that based on α, β values unifies several distance measures on SPD matrices and can learn new ones for a specific application.

![[math-ABDL calc.png]]

ABLD depends only on the generalized eigenvalues of X and Y. To compute it we use the formula below.

![[math-ABLD calc with eig.png]]

Where λ<sub>i</sub> denotes  the i-th eigenvalue of XY<sup>-1</sup>.

>[!important]
>Since λ<sub>i</sub> depends on input matrices, which are unpredictable, α and β are constrained to have the same sign.

> [!note]
> There are many math variations on calculations. Instead of subtracting -dlog(α+β) one can divide with (α+β).

| ABLD Property               | Explanation                                                                                                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Avoiding Degeneracy         | LogDet has to be positive definite and so α, β are constrained to have the same sign                                                                                        |
| Smoothness                  | With same sign α, β (except at origin α, β ->0) ABLD is smooth everywhere and we can develop Newton-type algorithms on them (discontinuity at the origin must be addressed) |
| Affine Invariance           | D<sup>(α,β)</sup> ( Χ \|\| Υ ) = D<sup>(α,β)</sup> ( AΧA<sup>T</sup> \|\| AYA<sup>T</sup> )                                                                                 |
| Dual Symmetry               | D<sup>(α,β)</sup> ( Χ \|\| Υ ) = D<sup>(β,α)</sup> ( Y \|\| X )                                                                                                             |
| Scaling invariance          | D<sup>(α,β)</sup> ( Χ \|\| Υ ) = D<sup>(α,β)</sup> ( cΧ \|\| cY )                                                                                                           |
| Intentity of Indiscernibles | D<sup>(α,β)</sup> ( Χ \|\| Υ ) = 0 only if X = Y                                                                                                                            |

## ABLD when α, β -> 0

![[Pasted image 20240417123205.png]]
## ABLD used in papers

| Distance Metric | α, β values  |
| --------------- | ------------ |
| BURG            | α=1, β=1     |
| AIRM            | α, β -> 0    |
| IDDL-S          | α!=β scalars |
| IDDL-V          | α==β vectors |
| IDDL-N          | α!=β vectors |

> [!important]
> Parameters α, β must always have the same sign except for the discontinuity case of α, β -> 0 . Making sure both are positive negates any sign issues.



