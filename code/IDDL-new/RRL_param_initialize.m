function params = RRL_param_initialize(num_atoms, train_labels)

    H = zeros(length(unique(train_labels)), length(train_labels));
    for z = 1:length(train_labels)
        H(train_labels(z), z) = 1;
    end

    W = ones(size(H,1), num_atoms);

    params.H = H;
    params.W = W;

end