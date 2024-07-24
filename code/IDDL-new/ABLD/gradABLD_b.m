function grad = gradABLD_b(X, Y, a, b)

    Z = X;
    X = Y;
    Y = Z;

    z = a;
    a = b;
    b = z;
    
    theta = a+b;
    ni = a*b;

    e = eig(X,Y);
    e(e<1e-10) = 1e-10;

    eab = a*e.^b + b*e.^(-a);

    A = (a*e.^b-ni*e.^(-a)*log(e))./(eab+eps);
    B = a/theta;
    C = log(eab/(theta+eps));
    grad = sum(A-B-C) / (a*ni);

end