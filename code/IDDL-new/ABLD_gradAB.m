function gg = ABLD_gradAB(d, num_atoms, n, X, B, ab, VW)

    E = zeros(d, num_atoms, n);
    parfor t=1:n           
        E(:,:,t) = cell2mat(cellfun(@(b) (eig(X{t}, b)), B, 'uniformoutput', false));            
    end
    E = permute(E,[1,3,2]);

    % safety for zero eigenvalues
    E(E<1e-10)=1e-10;
    lgE = log(E);    
    
    a = ab(1:num_atoms); %.alpha; 
    b = ab(num_atoms+1:end); %ab.beta;
    [gg_alpha, gg_beta]=deal(zeros(num_atoms,1));
    for k=1:num_atoms
        Eb = E(:,:,k).^(b(k)); 
        Ema = E(:,:,k).^(-a(k));
        EE = (a(k)*Eb + b(k)*Ema);
        
        CC =  log(a(k)+b(k)) - log(EE);
        gradV_a = ((a(k)*Eb - a(k)*b(k)*(Ema.*lgE(:,:,k)))./(EE+eps) -a(k)/(a(k)+b(k)+eps) + CC)/((a(k)^2)*b(k)+eps);
        % eig of spd matrixes that are inverted are just inverted eig
        gradV_b = ((b(k)*Ema + a(k)*b(k)*(Eb.*lgE(:,:,k)))./(EE+eps) -(b(k)/(a(k)+b(k)+eps)) + CC)/(a(k)*(b(k)^2)+eps);

        % dimentions are 1xN * Nx1
        gg_alpha(k) = sum(gradV_a,1)*VW(:,k);
        gg_beta(k) = sum(gradV_b,1)*VW(:,k);
    end

    gg = [gg_alpha; gg_beta];
end