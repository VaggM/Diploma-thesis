function graddiv = gradDiv_B_AIRM(B,isqrtX)
   nB = isqrtX*(B*isqrtX);
   [u,e] = schur(nB); 
   e=diag(e); 
%    e(e<1e-10)=1; 
   nB = u*diag(log(e+1e-5)./(e+1e-5))*u';
   graddiv = 2*isqrtX*(nB*isqrtX);
end