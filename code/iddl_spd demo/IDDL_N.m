% Author: Anoop Cherian
% Learning Discriminative AB divergence for SPD matrices.
%
% solve discriminative AB-log-det divergence with dictionary atoms
% here we will assume alpha \neq beta.
% Further, this code assume that alpha and beta are vectors. That is, they
% are equal, but there is one alpha and beta for every dictionary atom.
% However, we will also assume that alpha and beta have the same sign.
%
function [B, alphabeta, W, V] = IDDL_N(X_train, X_test, train_labels, test_labels, params)
    lam = params.lam;
    num_classes = numel(unique(train_labels));
    % Convert to cell from array
    X = cell(size(X_train,3),1);
    for z = 1:size(X_train,3)
        X{z} = squeeze(X_train(:,:,z));
    end
    Xtest = cell(size(X_test,3),1);
    for z = 1:size(X_test,3)
        Xtest{z} = squeeze(X_test(:,:,z));
    end
    num_atoms = params.num_atoms_per_class * num_classes;
    H = generate_binary_class_matrix(train_labels); 
    B_init = initialize_atoms(X, train_labels, num_atoms)';
    alpha_init = ones(num_atoms,1);
    beta_init = ones(num_atoms,1);
    n = length(X); d = size(X{1},1); 
    W = randn(size(H,1), num_atoms); W = ones(size(W));
    V = zeros(num_atoms, n);
    B = B_init;      
    alpha = alpha_init; beta=beta_init;
    
    % preprocessing to speed things up!    
    [sqX, isqX, iX] = deal(cell(length(X),1));
    for tt=1:length(X)
		X{tt} = X{tt} + 1e-20*eye(d);
        iX{tt} = inv(X{tt});
        sqX{tt} = sqrtm(X{tt});
        isqX{tt} = inv(sqX{tt});
    end
    % Set Up manifold structures
    spd_manifold = powermanifold(sympositivedefinitefactory(d), num_atoms);
    spd_problem.M = spd_manifold; 
    spd_problem.cost = @spd_objective;
    spd_problem.egrad =@spd_gradient;   
    manifold = euclideanfactory(2*num_atoms,1);
    alphabeta_problem.M = manifold; %productmanifold(manifold);    
    alphabeta_problem.cost = @alphabeta_objective;
    alphabeta_problem.egrad = @alphabeta_gradient;
          
    % internal common functions
    function obj = objective(B, W, alpha, beta)
        parfor t=1:n
            for k=1:num_atoms
                V(k,t) = abdiv(X{t}, B{k}, alpha(k), beta(k));
            end
        end
        obj = 0.5*sum(sum((H - W*V).^2)) + lam*sum(sum(W.^2));
    end

    % alpha manifold functionals, here we assume alpha = beta.
    function obj = alphabeta_objective(ab)
        obj = objective(B, W, ab(1:num_atoms), ab(num_atoms+1:end));
    end

    function gg = alphabeta_gradient(ab)
        M = W'*W; VW = V'*M-H'*W;
      
        E = zeros(d, num_atoms, n);
        parfor t=1:n           
            E(:,:,t) = cell2mat(cellfun(@(b) (eig(X{t}, b)), B, 'uniformoutput', false));            
        end
        E = permute(E,[1,3,2]);

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
            gradV_b = ((b(k)*Ema + a(k)*b(k)*(Eb.*lgE(:,:,k)))./(EE+eps) -(b(k)/(a(k)+b(k)+eps)) + CC)/(a(k)*(b(k)^2)+eps);

            gg_alpha(k) = sum(gradV_a,1)*VW(:,k);
            gg_beta(k) = sum(gradV_b,1)*VW(:,k);
        end
        gg = [gg_alpha; gg_beta];
        gg(isnan(gg)) = 0;
    end

    % spd manifold functionals
    function obj = spd_objective(B) 
        obj = objective(B, W, alpha, beta);
    end

    function gB = spd_gradient(B)
        M = W'*W; VW = V'*M-H'*W; %VM = V'*M; HW = H'*W; 
        gB = cell(num_atoms,1);
        for k=1:num_atoms
            gB{k} = 0;
            Bk = B{k}; iBk = inv(B{k}+eye(d)*1e-5);
            for i=1:n             
                gB{k} = gB{k} + VW(i,k)*gradBk(isqX{i}, sqX{i}, Bk, iBk, alpha(k), beta(k));
                %gradBk(X{i}, B{k}, alpha(k), beta(k));
            end  
        end
    end

    % weight matrix functionals.
    function W = update_W()
        W = (V*V'+2*lam*eye(num_atoms))\(V*H');
        W=W';
    end

     function [train_acc, test_acc] = class_accuracy()        
        [~, pred] = max(W*V, [], 1);
        [~, gt] = max(H, [], 1);
        train_acc = nnz(pred == gt)/numel(gt);
        
        Vt = zeros(num_atoms, length(Xtest));
        for tt=1:length(Xtest)
%             iXtest = inv(Xtest{tt});
            parfor kk=1:num_atoms
                Vt(kk,tt) = abdiv(Xtest{tt}, B{kk}, alpha(kk), beta(kk));
            end
        end
        [~, pred] = max(W*Vt,[],1);        
        test_acc = nnz(pred == test_labels')/numel(test_labels); 
    end  

    % main iterations.
    opts.maxiter = 20; % max num iterations of conjugate gradient.
    opts.verbosity = 0;
    objective(B, W, alpha,beta);
    W = update_W();
    [~, acc_test] = class_accuracy();
    fprintf('Initialization : Test Accuracy IDDL_N = %0.2f\n', acc_test);
    fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    alphabeta = [alpha;beta];
    for iter = 1:params.iter
        fprintf('Updating Atoms...\n')
        B = conjugategradient(spd_problem, B, opts);
        fprintf('Updating Divergence Parameters...\n')
        alphabeta = conjugategradient(alphabeta_problem, alphabeta, opts);
        alpha = alphabeta(1:num_atoms);
        beta = alphabeta(num_atoms+1:end);
        fprintf('mean alpha = %0.3f | mean beta = %0.3f\n', mean(alpha), mean(beta))
        if any(isnan([alpha;beta]))
            fprintf('matrices are not conditioned properly... nans in alpha, beta\n');
            break;
        end      
        fprintf('Updating Classifier...\n')
        W = update_W();      
        [~, acc_test] = class_accuracy();
        fprintf('iter = %d : Test Accuracy IDDL_N = %0.2f\n', iter, acc_test);
        fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    end
    alphabeta = [alpha;beta];
end
