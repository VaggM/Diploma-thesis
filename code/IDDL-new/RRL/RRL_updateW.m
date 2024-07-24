function W = RRL_updateW(H, V, gamma)

    W = (H*V')\(V'*V)-H*V'/gamma;
    % W = H*V'\(V*V'-gamma*Id)';
    W = W';

end