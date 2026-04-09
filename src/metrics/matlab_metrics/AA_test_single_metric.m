current = fileparts(mfilename('fullpath'));
addpath(genpath(current))
clc;

img1 = imread('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/3015.png');
img2 = imread('data/AANLIB/MyDatasets/SPECT-MRI/test/SPECT/3015.png');
img_f = imread('data/Fused_results/SPECT-MRI/ASFE-Fusion/3015.png');

if size(img1,3)>2, img1 = rgb2gray(img1); end
if size(img2,3)>2, img2 = rgb2gray(img2); end
if size(img_f,3)>2, img_f = rgb2gray(img_f); end

[s1, s2] = size(img1);
grey_level = 256;

img1_int = img1;
img2_int = img2;
img_f_int = img_f;

img1_float = im2double(img1)*255.0;
img2_float = im2double(img2)*255.0;
img_f_float = im2double(img_f)*255.0;

imgSeq = cat(3, img1_float, img2_float);

fprintf('Image size: %dx%d\n', s1, s2);

SSIM = SSIM_metrics(img1_float, img2_float, img_f_float);
fprintf('SSIM: %.6f\n', SSIM);
