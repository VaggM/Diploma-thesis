Key math needed to perform Ridge Regression Loss on IDDL.

## Matrices information

| Matrix     | Explanation                                                     |
| ---------- | --------------------------------------------------------------- |
| X \[dxdxN] | Input SPD data of N samples                                     |
| H \[LxN]   | One-off matrix based on class label for every Xi                |
| V \[nxN]   | Encoding for every Xi                                           |
| W \[Lxn]   | Weight to transform V into a probability matrix for every class |
**i** refers to one sample
**k** refers to one atom

## Objective function

![[math-RRL.png]]
## Parameter Updating

#### Dictionary B

Calculate gradient of dictionary B by

![[gradB calc]]
where $\theta_k = \alpha_k + \beta_k$ ,  $r_k = \frac{\alpha_k}{\beta_k}$ , $Z_i=X_i^{-1}$ and $\zeta_i = -(h_i-Wv_i)^{T}W$

> [!note]
> Dictionary B are calculated in the SPD plane powered to the number of atoms.

#### Parameters α,β

Calculate gradient of α by
![[math-grad a.png]]
where λ<sub>j</sub> refers to the j-th generalized eigenvalue of $X_i B_k^{-1}$.
Gradient of β is calculated based on dual symmetry if we replace:
- α -> β
- β -> α

> [!note]
> ABLD parameters are calculated in the euclidean plane.

#### Classifier W

 $W^{*} = HV^{T}(VV^{T}+\gamma I_d)^{-1}$

## Algorithm

**Input:** X, H, n
1) Initialize B and ABLD parameters
2) repeat some iterations of:
   - update B
   - update ABLD parameters
   - calculate W

> [!note]
> 1) Parallel computing is recommended for functions used in any gradient descends.
> 2) initializing B usually consists of random samples from data based on atoms per  class
> 3) it is recommended to initialize ABLD parameters with the Burg metric (α=1, β=1)

> [!important]
> ABLD is non-smooth at the origin α, β -> 0. We resort to the limit of the divergence, which is the AIRM.
> 
>![[math-ABLD origin.png]]
>
>And using ridge regression we calculate the gradient of dictionary B as follows.
>
>![[math-grad B origin.png]]
>![[math-grad B origin 2.png]]






