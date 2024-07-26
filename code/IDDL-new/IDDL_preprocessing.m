function [sqX, isqX, iX] = IDDL_preprocessing(X, d)
    [sqX, isqX, iX] = deal(cell(length(X),1));
    for i=1:length(X)
        % prevent issues
		X{i} = X{i} + 1e-20*eye(d);
        iX{i} = inv(X{i});
        sqX{i} = sqrtm(X{i});
        isqX{i} = inv(sqX{i});
    end
end