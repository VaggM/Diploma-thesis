function [BB, W, V] = IDDL_BURG(X_train, X_test, train_labels, test_labels, params)
    % This script performs a discriminative atom update for a set of SPD atoms 
    % of a dictionary or a set of SPD representatives. Furthermore an
    % additional performance evaluation based on the different selected
    % baselines is computed. 
    % The optimization problem governing the discriminative updates on the
    % atoms of the dictionary deploys a block coordinate descent scheme on the
    % following formulation based on ridge regression:
    %    minimize_{B_k, W, alpha, beta} ||H - W * V||_F^2 + lambda * ||W||_F^2
    %               subject to :          B_k > 0 
    %                            alpha * beta > 0
    % where: 
    %  H -- is a (# of classes) x (# of training samples) for which every column
    %       is the 1-bit class encoding for every sample. 
    %  W -- is a (# of classes) x (# of atoms) matrix.
    %  V -- is a (# of atoms/representatives) x (# of training samples) such
    %       that every column is computed as:
    %                                         v_i(k) = D^2(B_k||X_i)
    %                                         X_i is SPD training sample i
    %                                         D(C_i||C_j) is a selected
    %                                         divergence measuring the
    %                                         dissimilarity between matrices
    %                                         C_i and C_j
    % INPUT: params -- params.max_iter: is the maximum number of iterations for
    %                                   the whole block coordinate scheme
    %               -- params.atoms:    is the number of atoms allowed for
    %                                   every class of the classification 
    %                                   problem
    %               -- params.comps:    a set of the baselines that need to be
    %                                   computed for comparison purposes
    %               -- params.regs:     a collection of the regularizers for
    %                                   the different baselines for lambda
    %      : dataset:                   is the filename to the file that 
    %                                   contains the training and testing 
    %                                   data with gt_truth in the following
    %                                   format: 
    %                                      X_train is a (d)x(d)x(#trainsamples)
    %                                      X_test is a (d)x(d)x(#testsampels)
    %                                      train_labels (#trainsamples)x 1
    %                                      test_labels (#trainsamples)x 1
    %      
    % Panagiotis Stanitsas & Anoop Cherian
    % University fo Minnesota
    % February 2017
    %% Build H matrix of binary encodings
    num_classes = numel(unique(train_labels));
    H = generate_binary_class_matrix(train_labels);
    %% Main iterators over randomization and num_atoms
    %% Parameter Setting
    % Compute initializations of the Gallery atoms
    num_atoms = params.num_atoms_per_class * num_classes;
    BB = cell(num_atoms,1);
    atom = 1;
    for i = 1:num_classes
        inds_in = find(train_labels == i);
        inds_keep = randsample(inds_in,params.num_atoms_per_class);
        for j = 1:params.num_atoms_per_class
            BB{atom} = X_train(:,:,inds_keep(j));
            atom = atom + 1;
        end
    end
    %% Initialize W matrix as the solution to ||H - W * V||^2 + lambda2 * || W ||^2
    % Preliminary encodings
    a_k = [1; 1];
    V = zeros(num_atoms,size(X_train,3));
    Sk = cell(num_atoms, size(X_train,3));
    % initialize iX 
    iX = zeros(size(X_train));
    for i = 1:size(X_train,3)
        for cent = 1:num_atoms
            [u,e] = schur(X_train(:,:,i)); 
            e = diag(e); 
            e(e<0)=0;
            iX(:,:,i) = u*diag(1./(e+1e-7))*u';
            V(cent,i) = compute_diva_b(iX(:,:,i) * BB{cent},a_k);
            Sk{cent,i} = squeeze(iX(:,:,i)) * BB{cent};
        end
    end
    W = ((V * V' + params.lam * eye(size(V,1))) \ V * H')';
    [~, acc_test] = perf_check_ABLD(X_train, X_test, BB, train_labels, test_labels, W, a_k);
    fprintf('Initialization : Test Accuracy IDDL_BURG = %0.2f\n', acc_test);
    fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    %% Optimization Loop
    for iter = 1:params.iter
        fprintf('Updating Atoms...\n')
        BB = Update_BB_ABLD(X_train,W, BB, params.lam, H, a_k);
        fprintf('Updating Classifier...\n')
        W = Update_W_ABLD(X_train, BB, params.lam, H, a_k);
        [~, acc_test] = perf_check_ABLD(X_train, X_test, BB, train_labels, test_labels, W, a_k);
        fprintf('iter = %d : Test Accuracy IDDL_BURG = %0.2f\n', iter, acc_test);
        fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    end
end

