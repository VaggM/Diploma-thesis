% Author: Anoop Cherian
% Learning Discriminative AB divergence for SPD matrices.
%
% solve discriminative AB-log-det divergence with dictionary atoms
% here we will assume alpha \neq beta.
% Further, this code assume that alpha and beta are vectors. That is, they
% are equal, but there is one alpha and beta for every dictionary atom.
% However, we will also assume that alpha and beta have the same sign.
%
function [B, alphabeta, loss_params, V] = IDDL(X_train, X_test, train_labels, test_labels, params)
    lam = params.lam;
    loss_function = params.loss_func;
    loss_gradV = params.loss_func_gradV;
    %loss_params = params.loss_params;
    loss_param_init = params.loss_param_initialize;
    loss_param_update = params.loss_param_update;
    loss_class_accuracy = params.loss_class_accuracy;
    
    
    num_classes = numel(unique(train_labels));
    % Convert to cell from array
    % squeeze is used to change dxdx1 to dxd for X{z}
    X = cell(size(X_train,3),1);
    for z = 1:size(X_train,3)
        X{z} = squeeze(X_train(:,:,z));
    end
    Xtest = cell(size(X_test,3),1);
    for z = 1:size(X_test,3)
        Xtest{z} = squeeze(X_test(:,:,z));
    end

    % sizes
    n = length(X);
    d = size(X{1},1); 

    % preprocessing to speed things up!    
    [sqX, isqX, iX] = IDDL_preprocessing(X,d);

    % can change
    num_atoms = params.num_atoms_per_class * num_classes;

    % init B
    B = IDDL_initialize_atoms(X, train_labels, num_atoms)';
    
    % init ab
    alpha = ones(num_atoms,1);
    beta = ones(num_atoms,1);
    alphabeta = [alpha; beta];
    
    % Set Up manifold structures
    % spd manifold powered to the number of atoms
    spd_manifold = powermanifold(sympositivedefinitefactory(d), num_atoms);
    spd_problem.M = spd_manifold; 
    spd_problem.cost = @spd_objective;
    spd_problem.egrad =@spd_gradient; 

    % eucl manifold for vector [a; b]
    manifold = euclideanfactory(2*num_atoms,1);
    alphabeta_problem.M = manifold; %productmanifold(manifold);    
    alphabeta_problem.cost = @alphabeta_objective;
    alphabeta_problem.egrad = @alphabeta_gradient;
          
    % internal common functions
    % vector V is updated every time the objective is calculated
    function obj = objective(B, alpha, beta)
        V = IDDL_updateV(V, X, B, alpha, beta, n, num_atoms);
        obj = loss_function(V, lam, loss_params);
    end

    % alpha manifold functionals
    function obj = alphabeta_objective(ab)
        obj = objective(B, ab(1:num_atoms), ab(num_atoms+1:end));
    end

    function gg = alphabeta_gradient(ab)
        % dim Nxn
        VW = loss_gradV(V, loss_params);
        gg = ABLD_gradAB(d, num_atoms, n, X, B, ab, VW);

        %prevent nan
        gg(isnan(gg)) = 0;
    end

    % spd manifold functionals
    function obj = spd_objective(B) 
        obj = objective(B, alpha, beta);
    end

    function gB = spd_gradient(B)
        VW = loss_gradV(V, loss_params); 
        gB = ABLD_gradB(num_atoms, d, n, B, VW, isqX, sqX, alpha, beta);
    end

    % params init
    loss_params = loss_param_init(num_atoms, train_labels);

    V = zeros(num_atoms, n);
    V = IDDL_updateV(V, X, B, alpha, beta, n, num_atoms);

    % run objective to init V
    loss_params = loss_param_update(V, num_atoms, lam, loss_params);

    % messages
    [~, acc_test] = loss_class_accuracy(V, num_atoms, Xtest, B, alpha, beta, test_labels, loss_params);
    fprintf('Initialization : Test Accuracy IDDL_N = %0.2f\n', acc_test);
    fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

    % main iterations.
    opts.maxiter = 20; % max num iterations of conjugate gradient.
    opts.verbosity = 0;

    % Block Coordinate Descend
    for iter = 1:params.iter

        fprintf('Updating Atoms...\n')
        B = conjugategradient(spd_problem, B, opts);

        fprintf('Updating Divergence Parameters...\n')
        alphabeta = conjugategradient(alphabeta_problem, alphabeta, opts);
        alpha = alphabeta(1:num_atoms);
        beta = alphabeta(num_atoms+1:end);
        fprintf('mean alpha = %0.3f | mean beta = %0.3f\n', mean(alpha), mean(beta))
        % check for nan?!
        if any(isnan([alpha;beta]))
            fprintf('matrices are not conditioned properly... nans in alpha, beta\n');
            break;
        end

        fprintf('Updating Classifier...\n')
        loss_params = loss_param_update(V, num_atoms, lam, loss_params);

        [~, acc_test] = loss_class_accuracy(V, num_atoms, Xtest, B, alpha, beta, test_labels, loss_params);
        fprintf('iter = %d : Test Accuracy IDDL_N = %0.2f\n', iter, acc_test);
        fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    end
end
