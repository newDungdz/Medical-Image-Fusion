current = fileparts(mfilename('fullpath'));
addpath(genpath(current))
clc;

img1 = imread('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png');
img2 = imread('data/AANLIB/MyDatasets/SPECT-MRI/test/SPECT/4010.png');
img_f = imread('data/Fused_results/SPECT-MRI/ASFE-Fusion/4010.png');


% img1 = uint8([
%     82 40 20 85;
%     80 38 22 83;
%     78 36 24 81;
%     80 37 23 82;
% ]);

% img2 = uint8([
%     28 90 120 30;
%     26 88 125 28;
%     25 85 130 27;
%     26 87 128 28;
% ]);

% img_f = uint8([
%     55 65 70 58;
%     53 63 74 56;
%     52 60 78 55;
%     53 62 76 56;
% ]);

% test_matrix = [
%      3   120   43   156   119   145   241;
%    248   106   65   195   182   217   175;
%     64    17  151    97    23   157    34;
%    102   192  205   184    14   216   228;
%    199    20  190   109   215   105    80;
%    135   232   22   215    43   128   229;
%    195   203  116   147   174   213   113
% ];

% test_edge = edge(test_matrix);

% disp(test_edge);

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


fprintf('Image size: %dx%d\n', s1, s2);
disp("SSIM Offical")
disp(ssim(img1_float, img_f_float))
disp("SSIM Index")
disp(ssim_index(img1_float, img_f_float))
disp("Q")
disp(Peilla_metrics(img1,img2,img_f,1))
disp("Qw")
disp(Peilla_metrics(img1,img2,img_f,2))
disp("Qe")
disp(Peilla_metrics(img1,img2,img_f,3))