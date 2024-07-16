function W = Update_W_ABLD(X, BB, lam2, H, a_k)
    num_atoms = length(BB); 
    V = zeros(num_atoms,size(X,3));
    for i = 1:size(X,3)
        for cent = 1:num_atoms
            [u,e] = schur(X(:,:,i)); 
            e = diag(e); 
            e(e<0)=0;
            iX = u*diag(1./(e+1e-7))*u';
            V(cent,i) = compute_diva_b(iX * BB{cent},a_k);
            
        end
    end
   W = ((V * V' + lam2 * eye(size(V,1))) \ V * H')'; 
end

