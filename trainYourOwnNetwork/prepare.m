% ******Caution***** 
% You need to run this code if and only if you are trying to train your own Neural Network. 
% This Code reads the EMNIST dataset and convert the dataset as variables 
% suitable for "nprtool". It is supposed that you already have download the
% EMNST Dataset from the url https://www.kaggle.com/crawford/emnist and
% kept it in same directory where this code is present.
% After running this code two variables, named as 'targetsd' and 'inputs' 
% make sure you save them as .mat file
close all
clear
clc
% It allows to access file from another folder in the same directory
addpath(genpath('C:\Users\Akshay\Documents\MATLAB\MatlabCentralUpload\prepare\emnist'));
% Here emnist-balanced-train dataset is read
tr = csvread('emnist-balanced-train.csv', 1, 0);                  % read train.csv
n = size(tr, 1);                    % number of samples in the dataset
targets  = tr(:,1);                 % 1st column is |label|
targets(targets == 0) = 10;         % use '10' to present '0'
targetsd = dummyvar(targets);       % convert label into a dummy variable
inputs = tr(:,2:end);               % the rest of columns are predictors

inputs = inputs';                   % transpose input
targets = targets';                 % transpose target
targetsd = targetsd';               % transpose dummy variable