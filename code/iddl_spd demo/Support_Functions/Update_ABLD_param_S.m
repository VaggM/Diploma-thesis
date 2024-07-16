function param = Update_ABLD_param_S(X, a_init, W, lam2, BB, H)
    % compute inverses of X
    persistent iX;
    if isempty(iX)
        iX = zeros(size(X));
        for jj=1:size(X,3)
            [u,e] = schur(X(:,:,jj)); e = diag(e); e(e<0)=0;
            iX(:,:,jj) = u*diag(1./(e+1e-7))*u';
        end
    end
    num_atoms = length(BB);
    % Compute Sk's
    Sk = cell(num_atoms, size(X,3));
    for jj = 1:num_atoms
        for zz = 1:size(X,3)
            Sk{jj,zz} = squeeze(iX(:,:,zz)) * BB{jj};
        end
    end
    M = W' * W;   
    out = spg_general(Sk,num_atoms, H, W, M, lam2, a_init,10);
    param = out.x;
end


