function V = IDDL_updateV(V, X, B, alpha, beta, n, num_atoms)
    parfor t=1:n
        for k=1:num_atoms
            V(k,t) = ABLD(X{t}, B{k}, alpha(k), beta(k))
        end
    end
end