current = fileparts(mfilename('fullpath'));
addpath(genpath(current))
clc;

img1 = imread('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png');
img2 = imread('data/AANLIB/MyDatasets/SPECT-MRI/test/SPECT/4010.png');
img_f = imread('data/Fused_results/SPECT-MRI/ASFE-Fusion/4010.png');


% % ========================
% % TEST MATRIX 3x3
% % ========================
% img1 = uint8([80  20  85;
%               75  25  78;
%               80  22  88]);

% img2 = uint8([30 110  35;
%               28 120  32;
%               26 115  30]);

% img_f = uint8([58  70  60;
%                55  78  57;
%                62  75  61]);


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

scd_value = SCD_metrics(img1_float, img2_float, img_f_float);
fprintf('SCD: %.6f\n', scd_value);