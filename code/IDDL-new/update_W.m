function W = update_W(H,V,lam,num_atoms)
    W = (V*V'+2*lam*eye(num_atoms))\(V*H');
    W = W';
end