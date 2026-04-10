% -------------------------------
% Contourlet Transform Decomposition
% -------------------------------

clc; clear;

% Read image
img = imread('test.png');
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Parameters
pfilt = '9-7';      % Pyramid filter
dfilt = 'pkva';     % Directional filter
nlevs = [0 2 3];    % Number of directional subbands at each scale

% Decomposition
y = pdfbdec(img, pfilt, dfilt, nlevs);

% Structure:
% y{1} = lowpass
% y{2}, y{3}, ... = directional subbands (cell arrays)

disp('CT decomposition done');