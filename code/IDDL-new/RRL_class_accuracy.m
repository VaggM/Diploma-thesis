function [train_acc, test_acc] = RRL_class_accuracy(V, num_atoms, Xtest, B, alpha, beta, test_labels, params)
    W = params.W;
    H = params.H;
    
    [~, pred] = max(W*V, [], 1);
    [~, gt] = max(H, [], 1);
    train_acc = nnz(pred == gt)/numel(gt);
    
    Vt = zeros(num_atoms, length(Xtest));
    for tt=1:length(Xtest)
%             iXtest = inv(Xtest{tt});
        parfor kk=1:num_atoms
            Vt(kk,tt) = ABLD(Xtest{tt}, B{kk}, alpha(kk), beta(kk));
        end
    end
    [~, pred] = max(W*Vt,[],1);        
    test_acc = nnz(pred == test_labels')/numel(test_labels); 

    
end

