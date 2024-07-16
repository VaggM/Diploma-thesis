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