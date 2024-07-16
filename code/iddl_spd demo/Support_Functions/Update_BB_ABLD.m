function BB = Update_BB_ABLD(X, W, BB_init,lam2, H, a_k)
    persistent isqrtX
    persistent sqrtX
    persistent invX;
    if isempty(isqrtX)
        isqrtX = zeros(size(X));
        sqrtX = zeros(size(X));
        invX = zeros(size(X));
        for jj=1:size(X,3)
            [u_inter,e] = schur(X(:,:,jj)); e = diag(e); e(e<0)=0;
            isqrtX(:,:,jj) = u_inter *diag(1./(sqrt(e)+1e-7))*u_inter';
            sqrtX(:,:,jj) = u_inter*diag((sqrt(e)+1e-7))*u_inter';
            invX(:,:,jj) = u_inter*diag(1./(e+1e-7))*u_inter';
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
        parfor i = 1:size(X,3)
            for cent = 1:num_atoms
                V(cent,i) = compute_diva_b(invX(:,:,i) * BB{cent},a_k);
            end
        end
        obj = norm((H - W * V),'fro')^2 + lam2 * norm(W,'fro')^2;
    end

    function g = BB_gradient(BB)
        V = zeros(num_atoms,size(X,3));
        parfor i = 1:size(X,3)
            for cent = 1:num_atoms
                V(cent,i) = compute_diva_b(invX(:,:,i) * BB{cent},a_k);
            end
        end
        iB = cell(size(BB));
        for i = 1:num_atoms
            [u_in,e_in] = schur(BB{i});
            e_in = diag(e_in);
            iB{i} = u_in * diag(1./(e_in + 1e-7)) * u_in';       
        end      
        g = cell(length(BB),1);  
        for k = 1:num_atoms
            gg = 0;
            parfor z = 1:size(isqrtX,3)
                gg = gg + 2 * gradDiv_B(BB{k},  iB{k}, isqrtX(:,:,z), sqrtX(:,:,z), a_k)...
                    * (M(k,:) * V(:,z) - W(:,k)' * H(:,z));
            end
            g{k} = gg;
        end
    end
    options.maxiter = 20; 
    options.verbosity = 0;
    BB = conjugategradient(problem, BB_init, options);
end
