%% Handwritten Character Recognition Using Neural Networks
close all
clear
clc
addpath(genpath('C:\Users\Akshay\Documents\MATLAB\MiniProject\testImages'))
load('network.mat');
%% Input Image
imagePath = fullfile('testImages','akshay.png');
originalImage = imread(imagePath);
% Showing Original image
imshow(originalImage);
title('Original Image');
% Conversion to grayScale image
grayImage = rgb2gray(originalImage);
% Conversion to binary image
threshold = graythresh(grayImage);
binaryImage = ~im2bw(grayImage,threshold);
% Removes all object containing fewer than 30 pixels
moddedImage = bwareaopen(binaryImage,30);
pause(1)
% Showing binary image
figure(2);
imshow(moddedImage);
title('Modified Image');
% Labelling connected components
[L,Ne] = bwlabel(moddedImage);
% Measuring properties of image regions
propied = regionprops(L,'BoundingBox');
hold on
% Plot Bounding Box
for n=1:size(propied,1)
    rectangle('Position',propied(n).BoundingBox,'EdgeColor','g','LineWidth',2)
end
hold off
pause (1)
%% Image Segmentation
for n=1:Ne
    [r,c] = find(L==n);
    n1 = moddedImage(min(r):max(r),min(c):max(c));
    n1 = imresize(n1,[128 128]);
    n1 = imgaussfilt(double(n1),1);
    n1 = padarray(imresize(n1,[20 20],'bicubic'),[4 4],0,'both');
    fullFileName = fullfile('segmentedImages', sprintf('image%d.png', n));
    imwrite(n1, fullFileName);
    pause(1)
end
%% Feeding to Neural Network and Detected Text
for i=1:Ne
    segImage=reshape(double(imread(fullfile('segmentedImages', sprintf('image%d.png', i)))) , 784, 1);
    outputMatrix=net(segImage);
    row=find(ismember(outputMatrix, max(outputMatrix(:))));
    character = double(imread(fullfile('segmentedImages', sprintf('image%d.png', i))));
    detectedWord(1,i)=imageLabeler(row);
end
disp('Detected Text: ')
disp(detectedWord)