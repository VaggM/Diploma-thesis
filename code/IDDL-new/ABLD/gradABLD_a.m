function grad = gradABLD_a(X, Y, a, b)


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