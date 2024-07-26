function grad = RRL_gradV(V, params)
    
    H = params.H;
    W = params.W;

    grad = V'*(W'*W) - H'*W;

end