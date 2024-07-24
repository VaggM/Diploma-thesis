function grad = gradABLD_Y(X, Y, a, b)

    %% option 1
    r = a/b;
    theta = a+b;

    Z = inv(X);

    ZYZ = (Z^(1/2)*Y*Z^(1/2))^(theta);

    d = length(X);
    Id = diag(ones([d 1]));

    grad = r*thetta*inv(Y)*Z^(1/2)*ZYZ\(Id+r*ZYZ) - (1/b)*inv(Y);

    %% option 2
    % r = a/b;
    % theta = a+b;
    % Z = inv(X);
    % ZYZ = (Z^(1/2)*Y*Z^(1/2));
    % [U, D] = schur(ZYZ);
    % delta = diag(D);
    % 
    % delta = diag(delta.^theta ./ (1 + r*delta.^theta));
    % ZU = Z^(-1/2)*U;
    % grad = r*theta*inv(Y)*ZU*delta*inv(ZU)-(1/b)*inv(Y);

end