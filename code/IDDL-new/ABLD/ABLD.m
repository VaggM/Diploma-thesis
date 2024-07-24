function D = ABLD(X,Y,a,b)

    % try
    %     e = eig(X,B);  d=length(e); e(e<=1e-10)=1e-10;
    % catch
    %     disp('error!');
    %     B(isnan(B)) = 0; B = B + 1e-5*eye(size(B,1));
    %     [~,e] = schur(X\B); e=diag(e); d=length(e);
    % end

    e = eig(X,Y);
    % prevent zeros
    e(e<eps) = eps;
    d = length(e);

    % calc abld
    D = (sum(log(a*e.^b + b*e.^-a)) - d*log(a+b)) / (a*b+eps);

end