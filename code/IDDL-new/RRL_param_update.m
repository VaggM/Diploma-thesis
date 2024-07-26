function params = RRL_param_update(V, num_atoms, lam, params)

    H = params.H;
 
    W = (V*V'+2*lam*eye(num_atoms))\(V*H');
    W = W';

    params.W = W;
    
end