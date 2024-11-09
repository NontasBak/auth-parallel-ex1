clc, clearvars, close all;

% Load train_data.mat
load('train_data.mat', 'train_data');
load('neighbors_data.mat', 'neighbors_data');

% Parameters
n = 10000; % First 10k queries
m = size(train_data, 2); % Number of points in C
d = size(train_data, 1); % Dimensionality
k = 100; % Number of nearest neighbors

% Initialize C matrix
C = train_data';

% Initialize Q matrix (first 10k points from C)
Q = C(1:n, :);

% Use MATLAB's knnsearch for validation
[idx, dist] = knnsearch(C, Q, 'k', k);

% Save the indices of the nearest neighbors to knn_neighbors.mat
knn_neighbors = idx';
save('knn_neighbors.mat', 'knn_neighbors');

disp('MATLAB k-NN Indices:');
disp(idx);
disp('MATLAB k-NN Distances:');
disp(dist);
