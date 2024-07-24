function obj = RRL(H, W, V, lam)
    obj = 0.5*sum(sum((H - W*V).^2)) + lam*sum(sum(W.^2));
end