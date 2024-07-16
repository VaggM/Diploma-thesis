function BB = Update_BB_AIRM(X, W, BB_init,lam, H)
    persistent isqrtX;
    if isempty(isqrtX)
        isqrtX = zeros(size(X));
        for jj=1:size(X,3)
            [u,e] = schur(X(:,:,jj)); e = diag(e); e(e<0)=0;
            isqrtX(:,:,jj) = u*diag(1./(sqrt(e+1e-5)))*u';
        end
    end
    num_atoms = length(BB_init); 
    d = size(BB_init{1},1); 
    B_manifold = sympositivedefinitefactory(d);
    BB_manifold = powermanifold(B_manifold, num_atoms);    
    problem.M = BB_manifold;
    problem.cost = @BB_objective;
    problem.egrad = @BB_gradient;
    M = W' * W;
    function obj = BB_objective(BB)     
        V = zeros(num_atoms,size(X,3));
        parfor ii = 1:size(X,3)
            V(:,ii) = cellfun(@(xx) geodist(X(:,:,ii),xx), BB)';
        end
        obj = norm((H - W * V),'fro')^2 + lam * norm(W,'fro')^2;
    end

    function g = BB_gradient(BB)
        V = zeros(num_atoms,size(X,3));
        parfor ii = 1:size(X,3)
           for cent = 1:num_atoms
               V(cent,ii) = geodist(squeeze(X(:,:,ii)), BB{cent});
           end
        end
        g = cell(length(BB),1);       
        for k = 1:num_atoms
            gg = 0;
            parfor z = 1:size(isqrtX,3)
                gg = gg + 2 * gradDiv_B_AIRM(BB{k}, isqrtX(:,:,z)) * ...
                    (M(k,:) * V(:,z) - W(:,k)' * H(:,z));
            end
            g{k} = gg;
        end
    end
    options.maxiter = 20; 
    options.verbosity = 0;
    BB = conjugategradient(problem, BB_init, options);
end