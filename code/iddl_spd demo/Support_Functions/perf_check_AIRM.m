function [acc1,acc2] = perf_check_AIRM(X_train, X_test, BB, train_labels,test_labels,W)
    num_atoms = length(BB);
    v_k_new = zeros(num_atoms, size(X_train,3));
    v_test_new = zeros(num_atoms, size(X_test,3));
    parfor i = 1:size(X_train,3)
        for cent = 1:num_atoms
            v_k_new(cent,i) = geodist(squeeze(X_train(:,:,i)), BB{cent});
        end
    end
    parfor i = 1:size(X_test,3)
        for cent = 1:num_atoms
            v_test_new(cent,i) = geodist(squeeze(X_test(:,:,i)), BB{cent});
        end
    end
    train_mat = W * v_k_new;
    test_mat = W * v_test_new;
    [~,I] = max(train_mat,[],1);
    acc1 = 100 * sum(train_labels == I') / size(X_train,3);
    [~,I] = max(test_mat,[],1);
    acc2 = 100 * sum(test_labels == I') / size(X_test,3);
end