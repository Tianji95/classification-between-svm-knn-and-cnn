 function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE

[P,N] = size(X);
H=eye(P+1);
H(P+1,P+1)=0;
f=zeros(P+1,1);
X=[ones(1,N);X];

A = ones(P+1,N);
fprintf('before resize\n');
for i = 2:P+1
    A(i,:) = y.*X(i,:);
end
fprintf('after resize\n');
A = -A';
b = -ones(N,1);
options = optimoptions('quadprog',...
    'Algorithm','interior-point-convex',...
    'Display','off');
w = quadprog(H,f,A,b,...
    [],[],[],[],[],options);
num = 0;
fprintf('before count\n');
for i=1:N
    if (w'*X(:,i)<=1)
        num = num+1;
    end
end
fprintf('after count\n');
end
