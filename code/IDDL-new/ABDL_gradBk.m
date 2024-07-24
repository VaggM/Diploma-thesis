function grad = ABDL_gradBk(H, W, V)
    % M = W'*W; VW = V'*M-H'*W;
    grad = V'*(W'*W) - H'*W;

end