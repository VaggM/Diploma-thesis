function obj = RRL(V, lam, params)
    H = params.H;
    W = params.W;
    obj = 0.5*sum(sum((H - W*V).^2)) + lam*sum(sum(W.^2));
end