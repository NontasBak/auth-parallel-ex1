clc, clearvars, close all;

n = 5;
m = 10;
d = 2;
k = 3;

% Initialize C matrix
C = zeros(m, d);
for i = 1:m
    for j = 1:d
        C(i, j) = i - 1 + j - 1;
    end
end

% Initialize Q matrix
Q = zeros(n, d);
for i = 1:n
    for j = 1:d
        Q(i, j) = i - 1 + j - 1;
    end
end

% Display matrices
disp('C matrix:');
disp(C);
disp('Q matrix:');
disp(Q);

% Call the knn function with C and k
knn_validate(C, Q, k);

function knn_validate(C, Q, k)
    D = sqrt(bsxfun(@plus, sum(C.^2, 2), sum(Q.^2, 2)') - 2 * (C * Q'));

    % Display the distance matrix
    disp('Distance matrix D:');
    disp(D);
    
    % Use MATLAB's knnsearch for validation
    [idx, dist] = knnsearch(C, Q, 'k', k);
    
    disp('MATLAB k-NN Indices:');
    disp(idx);
    disp('MATLAB k-NN Distances:');
    disp(dist);
end
