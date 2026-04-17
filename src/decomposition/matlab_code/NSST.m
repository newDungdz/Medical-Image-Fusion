% -------------------------------
% NSST Decomposition (Easley version)
% -------------------------------

current = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(current, 'nsst_toolbox')));

clc; clear;

% Read image
img = imread('test.png');

% img = [10   30  200  200;
%        10   30  200  200;
%        10  100  200  200;
%        10  100  150  100];


if size(img,3) == 3
    img = rgb2gray(img);
end
img = im2double(img);

% Shear parameters
shear_parameters.dcomp = [2, 2, 2];
shear_parameters.dsize = [16, 16, 16];
% shear_parameters.dcomp = [1];   % 2 directions only
% shear_parameters.dsize = [2];   % very small filter
% Low-pass filter
lpfilt = '9-7';   % or gaussian if error

% Decomposition
[dst, shear_f] = nsst_dec1e(img, shear_parameters, lpfilt);

rec = nsst_rec1(dst, lpfilt);
error = norm(img(:) - rec(:)) / norm(img(:));
disp(['Reconstruction error: ', num2str(error)]);


disp('NSST decomposition done');
% % Visualize the decomposition
% figure;

% % Low-pass
% subplot(2,3,1);
% imshow(mat2gray(dst{1}));
% title('Low-pass');

% % Shear components (dst{2})
% ndir = size(dst{2}, 3);

% for k = 1:ndir
%     subplot(2,3,k+1);
%     imshow(mat2gray(dst{2}(:,:,k)));
%     title(sprintf('Shear Dir %d', k));
% end