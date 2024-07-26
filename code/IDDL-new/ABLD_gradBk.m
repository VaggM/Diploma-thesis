function gB = gradBk(isqX, sqX, B, iB, alpha, beta)
    %sqX = sqrtm(X); isqX=eye(size(B,1))/sqX; iB = eye(size(B,1))/B;
    S = (isqX*B)*isqX;
	try
	    [u,e]=schur(S); e=diag(e);
    catch
        disp('gradBk: exception caught');
		S(isnan(S)) = 0;
		[u,e]=eig(S); e=diag(e); e(e<=0)=1e-7;
	end
    uu = u*diag((e.^(alpha+beta))./(1+(alpha/beta)*(e.^(alpha+beta))))*u';
    % for somereason Cherian had -alpha*iB
    gB = ((alpha+beta)*alpha/(beta)*((iB*sqX) * uu)*isqX -iB/(beta+eps))/(alpha*beta+eps);
    
end