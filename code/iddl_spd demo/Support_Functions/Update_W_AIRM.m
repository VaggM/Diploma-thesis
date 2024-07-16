function W = Update_W_AIRM(X, BB, lam, H)
    num_atoms = length(BB); 
    V = zeros(num_atoms,size(X,3));
    parfor ii = 1:size(X,3)
        V(:,ii) = cellfun(@(xx) geodist(X(:,:,ii),xx), BB)';
    end
    W = ((V * V' + lam * eye(size(V,1))) \ V * H')';
end