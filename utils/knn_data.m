clc, clearvars, close all;

% Load "train" dataset from HDF5 file
train_data = h5read('../data/sift-128-euclidean.hdf5', '/train');

% Save "train" dataset to a .mat file
save('../data/train_data.mat', 'train_data');

% Load and print the contents of the .mat file
% loaded_data = load('../data/train_data.mat');
% disp('Contents of train_data.mat:');
% disp(loaded_data.train_data);
% disp(size(loaded_data.train_data));
% disp(ismember(5, loaded_data.train_data));


% % List all datasets in the HDF5 file
% info = h5info('sift-128-euclidean.hdf5');
% disp('Datasets in the HDF5 file:');
% for i = 1:length(info.Datasets)
%     disp(info.Datasets(i).Name);
% end

% % Load and print contents of each dataset
% datasets = {'train', 'test', 'distances', 'neighbors'};
% for i = 1:length(datasets)
%     data = h5read('sift-128-euclidean.hdf5', ['/' datasets{i}]);
%     disp(['Contents of ' datasets{i} ':']);
%     % disp(data);
%     disp(size(data));
% end