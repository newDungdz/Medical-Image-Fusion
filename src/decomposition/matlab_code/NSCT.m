% -------------------------------
% NSCT Decomposition
% -------------------------------

clc; clear;

% Read image
img = imread('test.png');
if size(img, 3) == 3
    img = rgb2gray(img);
end
% Parameters
nlevs = [2 3 4];     % Directions per scale
dfilt = 'dmaxflat7'; % Directional filter
pfilt = 'maxflat';   % Pyramid filter

% Decomposition
y = nsctdec(img, nlevs, dfilt, pfilt);

disp('NSCT decomposition done');