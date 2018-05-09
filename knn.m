function y = knn(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, P-by-N_test matrix.
%           X_train:   training sample features, P-by-N matrix.
%           y_train:   training sample labels, 1-by-N row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1-by-N_test row vector.
%

% YOUR CODE HERE
[P, N_test] = size(X);
[~, N] = size(X_train);
y = zeros(1,N_test);

for idx = 1:N_test
    dist_N = pdist2(X(:,idx)', X_train', 'euclidean');
    [~, sort_pos] = sort(dist_N);
    y_sort = y_train(sort_pos(1:K));
    unique_y = unique(y_sort);
    count_y = histc(y_sort,unique_y);
    [~,max_index] = max(count_y);
    y(1,idx)=unique_y(max_index);
end

end

