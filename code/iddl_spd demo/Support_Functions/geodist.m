function d = geodist(p,x)
    d = sum(log(eig(p,x)).^2);
end
