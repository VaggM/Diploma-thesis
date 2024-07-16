function H = generate_binary_class_matrix(labels)
u = unique(labels);
H = zeros(length(u), length(labels));
for t=1:length(labels)
H(labels(t),t) = 1;
end
end
