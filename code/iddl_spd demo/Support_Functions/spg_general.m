function out = spg_general(Sk,num_atoms, H, W, M, lam2, x0,mit)
   % Altered implementation for Spectral Projected Gradient descent
   % Panagiotis Stanitsas & Anoop Cherian
   % February 2017
   
   gfx  = @(x) eval_fun_grad(Sk,num_atoms, H, W, M, lam2,x);       
   prx  = @(x) proj(x);   
   options.verbose = 0;
   [out.x,out.f,out.funEvals,out.projects,out.itertime,out.info] = cleanSPG(gfx,x0,prx,mit,options);
end

function [f,g] = eval_fun_grad(Sk,num_atoms, H, W, M, lam2, a_b)
   % objective
   f = ak_objective(Sk, num_atoms, a_b, H, W, lam2);
   % gradient
   g = ak_gradient(Sk, num_atoms, a_b, H, W, M);
end

function obj = ak_objective(Sk, num_atoms, a_b, H, W, lam2)
    V = zeros(size(Sk));
    for ii = 1:num_atoms
        for l = 1:size(Sk,2)
               V(ii,l) = compute_diva_b(Sk{ii,l},a_b);
        end
    end
    obj = norm((H - W * V),'fro')^2 + lam2 * norm(W,'fro')^2;
end

function dist = compute_diva_b(Sk,a_b)
    alpha = a_b(1);
    beta = a_b(2);
    [~, e_in] = schur(Sk);
    e_in = diag(e_in);
    e_in(e_in<0) = 0;
    e_in = e_in + 1e-7;
    dist = 1 / (alpha * beta) * sum(log((alpha * e_in.^(beta) + beta * e_in.^(-alpha))...
        / (alpha + beta)));
end

function grad = ak_gradient(Sk, num_atoms, a, H, W, M)
    V = zeros(size(Sk));
    for pp = 1:num_atoms
        for p = 1:size(Sk,2)
               V(pp,p) = compute_diva_b(Sk{pp,p},a);
        end
    end
    grad = zeros(2,1);
    gg1 = 0;
    gg2 = 0;
    for z = 1:size(Sk,2)
        grad_vec = nabla_div(num_atoms, a, Sk(:,z));
        gg1 = gg1 - 2 * H(:,z)' * W * grad_vec(:,1) + 2 * V(:,z)' * M * grad_vec(:,1); 
        gg2 = gg2 - 2 * H(:,z)' * W * grad_vec(:,2) + 2 * V(:,z)' * M * grad_vec(:,2); 
    end   
    grad(1,1) = gg1;
    grad(2,1) = gg2;
end

function grad_vec = nabla_div(num_atoms, a_b, Sk_i)
   alpha = a_b(1);
   beta = a_b(2);
   grad_vec = zeros(num_atoms,2);
   for d = 1:num_atoms
        [~, e_in] = schur(Sk_i{d});
        e_in = diag(e_in);
        e_in(e_in <0) = 0;
        e_in = e_in + 1e-7;
        lam_a_b = e_in.^(alpha + beta);
        denom = alpha * beta * (alpha + beta) * (beta + alpha *lam_a_b);
        logdiv = log((beta * e_in.^(-alpha) + alpha * e_in.^(beta))/(alpha + beta));
        grad_vec(d,1) = sum((alpha * beta * (-1 + lam_a_b) - ...
            (alpha + beta) * (alpha * beta * log(e_in) + (beta + alpha * lam_a_b) .*...
            logdiv)) ./ (alpha * denom));
        grad_vec(d,2) = sum((-alpha * beta * (-1 + lam_a_b) + ...
            (alpha + beta) * (alpha * beta * lam_a_b .* log(e_in) - (beta + alpha * lam_a_b) .*...
            logdiv)) ./ (beta * denom));
   end
end

function M = proj(M)
   M = eye(size(M,1)) * M; 
end
