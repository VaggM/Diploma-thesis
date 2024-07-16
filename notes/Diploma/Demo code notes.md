
## IDDL-N

This is the most complex case where ABLD parameters are unequal vectors.

Algorithm steps
- initialize matrices X, H, B and parameters ABLD plus lambda (γ)
- calculate initial matrices V and W
- calculate $X^{-1}, X^{-1/2}, X^{1/2}$  (adding small esp to diagonal to avoid zeros)
- define the optimization structures
- for some iterations:
  1) update dictionary B
  2) update parameters α, β
  3) update classifier W

The inilization of the dictionary B is done by picking random samples per class based on how many atoms per class should consist it while the initialization of ABLD parameters is done by settings all elements to 1.

> [!note]
> The white papers are written based on IDDL-N and the code that Anoop Cherian used for it. So any math used in the paper is used directly in this IDDL variant while the other variants might use different math approaches to reach the same solution.
## Optimization Structures

| Updated Variable     | Dictionary B                                                                                                                          | Parameters α, β                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manifold Structure   | $SPD^{n}$ manifold plane                                                                                                              | $R^{2n}$ euclidean plane                                                                                                                                    |
| Optimization Cost    | RRL objective for current V                                                                                                           | RRL objective for current V                                                                                                                                 |
| Gradient Calculation | Each atom's gradient is a \[dxd] matrix and given by<br><br>$gradB_k = \zeta_{i,k} * 2 gradB_k( D^{\alpha_k,\beta_k}(X_i \|\| B_k) )$ | After calculating all eigenvalues of $X_i B_k^{-1}$ the gradient for each parameter on the current atom is given by<br>![[Pasted image 20240418093326.png]] |

> [!note]
> 1) Parameters α, β are updated with RCG although being in the euclidean plane
> 2) The α, β plane is in $R^{2n}$ cause its a vector of 2n elements (n for α and n for β)
> 3) Calculating current matrix V uses the eigenvalues of $X_i B_k^{-1}$ 
> 4) When calculating eigenvalues for ABLD gradients a tensor E is used in code consisting of multiple matrixes that present the eigenvalues calculated for each input for one atom at a time

> [!important]
> 1) In any division in calculations an $\epsilon->0$ is added to prevent dividing by zero
> 2) All eigenvalues have a lower bound  $\epsilon->0$
> 3) When calculating ABLD gradient we must make sure that the result is not nan
> 4) When passing ABLD parameters in RCG we must use α, β within one vector of 2n elements since the manopt function updates only one variable
> 5) In all codes parfor is used for doing parallel computations for RCG
> 6) Gradient for dictionary B in all demo codes doesn't use the calculation method with Schur Decomposition that speeds up the process

> [!warning]
> 1) Gradient for β is calculated using the dual symmetry property but in code the eigenvalues are still calculated from $X_i B_k^{-1}$ instead of $B_k X_i^{-1}$
> 2) The discontinuity case at the origin is not taken into account anywhere

Below there is a list of changes in code when calculating other IDDL variants.

## IDDL-V changes

This is the case where ABLD parameters are equal vectors.
Changes:
1) Gradient calculation math is different for ABLD parameters
## IDDL-S changes

This is the case where ABLD parameters are unequal scalars.
Changes:
1) Eigenvalues for ABLD are calculated by $X_i^{-1} B_k$ for some reason
2) When updating dictionary B the formula uses $gradB_k = \zeta_{i,k} * 2 gradB_k( D^{\alpha_k,\beta_k}(X_i || B_k) )$, where the 2 is unexpected
3) Code creates manifold every time the dictionary is updated making it inefficient
4) ABLD parameters are scalar and updated by Spectral Projected Gradient

> [!note]
> The Spectral Projected Gradient in code is a modified version from the one used by the manopt library.

## IDDL-Burg changes

This is the case where ABLD parameters are fixxed at α=1, β=1.
Changes:
1) When updating dictionary B the formula uses $formula$, where the 2 is unexpected
2) Code creates manifold every time the dictionary is updated making it inefficient

## IDDL-AIRM changes

This is the case where ABLD parameters are fixxed at the origin and AIRM is used.
Changes:
1) Computes geodist for V over ABLD
2) Uses AIRM to calculate the gradient of dictionary B
3) When updating dictionary B the formula uses $gradB_k = \zeta_{i,k} * 2 gradB_k( D^{\alpha_k,\beta_k}(X_i || B_k) )$, where the 2 is unexpected