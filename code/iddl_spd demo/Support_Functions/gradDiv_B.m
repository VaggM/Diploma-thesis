function graddiv = gradDiv_B(B, iB, isqrtX, sqrtX, a_k)
   alpha = a_k(1);
   beta = a_k(2);
   nB = isqrtX*(B*isqrtX);
   [u,e] = schur(nB); 
   e=diag(e);
   xBxpower = u * diag((e+1e-7).^(alpha + beta)) * u';
   comp = eye(size(B,1)) + alpha / beta * xBxpower;
   [u,e] = schur(comp);
   e = diag(e);
   invcomp = u * diag(1./(e+1e-7)) * u';
   graddiv = (alpha + beta) / beta^2 * (iB * sqrtX * xBxpower * invcomp * isqrtX) - 1 / beta * iB; 
end
