function param = gridsearch_alpha_beta(S_k, train_labels, lam, H, type, init, max_val)
% Performs a grid search in the parameter space of alpha beta divergence.
% Accommodates the cases that:
% 1. Learns an alpha = beta deloying a grid search for a random subset of
%    the dataset. -- type = 1;
% 2. alpha = beta and we seek for an update given a value in a predefined 
%    vicinity around the value that we have. -- type = 2;
% 3. Learns an alpha ~= beta deploying a grid search for a random subset
%    of the dataset. -- type = 3;
% 4. alpha ~= beta and we seek for an update imposing a small grid around a
%    value of interest (parameters from previous step). -- type = 4;
  
    num_samples = 100;
    ran = 0.5;
    switch type
        case 1
            rand_inds = randsample(size(S_k,2), num_samples);
            S_k = S_k(:,rand_inds);
            train_labels = train_labels(rand_inds);
            H = H(:,rand_inds);
            % Full grid search alpha = beta
            param_line = 0.1:0.1:max_val;
            accuracy = zeros(1,length(param_line));
            for i = 1:length(param_line)
                V = zeros(size(S_k));
                for sam = 1:size(S_k,2)
                    for cent = 1:size(S_k,1)
                        V(cent,sam) = compute_diva_b(S_k{cent,sam}, [param_line(i) param_line(i)]);
                    end
                end 
                W = ((V * V' + lam * eye(size(V,1))) \ V * H')';
                accuracy(i) = checkperformanceLinClass(V, train_labels, W);
            end
            % Return parametrization
            [~,ind] = max(accuracy);
            param = [param_line(ind); param_line(ind)];
        case 2
            rand_inds = randsample(size(S_k,2),num_samples);
            S_k = S_k(:,rand_inds);
            train_labels = train_labels(rand_inds);
            H = H(:,rand_inds);
            % Partial Update alpha = beta
            param_line = max([0.1 init(1)-ran]):0.1:min([max_val init(1)+ran]);
            accuracy = zeros(1,length(param_line));
            for i = 1:length(param_line)
                V = zeros(size(S_k));
                for sam = 1:size(S_k,2)
                    for cent = 1:size(S_k,1)
                        V(cent,sam) = compute_diva_b(S_k{cent,sam}, [param_line(i) param_line(i)]);
                    end
                end 
                W = ((V * V' + lam * eye(size(V,1))) \ V * H')';
                accuracy(i) = checkperformanceLinClass(V, train_labels, W);
            end
            % Return parametrization
            [~,ind] = max(accuracy);
            param = [param_line(ind); param_line(ind)];
        case 3
            rand_inds = randsample(size(S_k,2),num_samples);
            S_k = S_k(:,rand_inds);
            train_labels = train_labels(rand_inds);
            H = H(:,rand_inds);
            % Full grid search
            [a_grid, b_grid] = meshgrid(0.1:0.2:max_val);
            accuracy = zeros(size(a_grid));
            for i = 1:size(a_grid,1)
                for j = 1:size(a_grid,2)
                    V = zeros(size(S_k));
                    for sam = 1:size(S_k,2)
                        for cent = 1:size(S_k,1)
                            V(cent,sam) = compute_diva_b(S_k{cent,sam}, [a_grid(i,j) b_grid(i,j)]);
                        end
                    end 
                    W = ((V * V' + lam * eye(size(V,1))) \ V * H')';
                    accuracy(i,j) = checkperformanceLinClass(V, train_labels, W);
                end
            end   
            % Return parametrization of maximum accuracy
            ind = find(accuracy == max(max(accuracy)));
            [ind1,ind2] = ind2sub(size(accuracy),ind(1));
            param = [a_grid(ind1,ind2); b_grid(ind1,ind2)];         
        case 4
            rand_inds = randsample(size(S_k,2),num_samples);
            S_k = S_k(:,rand_inds);
            train_labels = train_labels(rand_inds);
            H = H(:,rand_inds);
            % Partial Update
            a_range = max([0.1 init(1)-ran]):0.1:min([max_val init(1)+ran]);
            b_range = max([0.1 init(2)-ran]):0.1:min([max_val init(2)+ran]);
            [a_grid, b_grid] = meshgrid(a_range, b_range);
            accuracy = zeros(length(b_range), length(a_range));
            for i = 1:length(a_range)
                for j = 1:length(b_range)
                    V = zeros(size(S_k));
                    for sam = 1:size(S_k,2)
                        for cent = 1:size(S_k,1)
                            V(cent,sam) = compute_diva_b(S_k{cent,sam}, [a_grid(i,j) b_grid(i,j)]);
                        end
                    end 
                    W = ((V * V' + lam * eye(size(V,1))) \ V * H')';
                    accuracy(i,j) = checkperformanceLinClass(V, train_labels, W);
                end
            end 
             % Return parametrization of maximum accuracy
            ind = find(accuracy == max(max(accuracy)));
            [ind1,ind2] = ind2sub(size(accuracy),ind);
            param = [a_grid(ind1(1),ind2(1)); b_grid(ind1(1),ind2(1))];        
    end
    % Function to compute accuracy on the training set
    function acc1 = checkperformanceLinClass(V, train_labels, W)
        train_mat = W * V;
        [~,I] = max(train_mat,[],1);
        acc1 = 100 * sum(train_labels == I') / size(V,2);     
    end
end