function gradB = ABLD_gradB(num_atoms, d, n, B, VW, isqX, sqX, alpha, beta)

    gradB = cell(num_atoms,1);
    for k=1:num_atoms
        gradB{k} = 0;
        % safety for zeros
        Bk = B{k}; 
        iBk = inv(Bk+eye(d)*1e-5);
        for i=1:n             
            gradB{k} = gradB{k} + VW(i,k)*ABLD_gradBk(isqX{i}, sqX{i}, Bk, iBk, alpha(k), beta(k));
        end  
    end

end