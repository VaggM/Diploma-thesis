function B_init = initialize_atoms(X_train, train_labels, num_atoms)
    num_classes = numel(unique(train_labels));
    B_init = cell(num_atoms,1);
    atom = 1;
    for i = 1:num_classes
        inds_in = find(train_labels == i);
        inds_keep = randsample(inds_in, num_atoms/num_classes);
        for j = 1:num_atoms/num_classes
            B_init{atom} = X_train{inds_keep(j)};
            atom = atom + 1;
        end
    end
end