clc;
params.dcomp = [2, 2, 2];
params.dsize = [16, 16, 16];


stats = @(x, name) fprintf([ ...
    '%s shape=(%d,%d) min=%.4f max=%.4f mean=%.4f energy=%.4f\n'], ...
    name, size(x,1), size(x,2), ...
    min(x(:)), max(x(:)), mean(x(:)), sum(x(:).^2));

% Read image (grayscale)
img = imread('test.png');
if size(img,3) == 3
    img = rgb2gray(img);
end
img = double(img);

% Parameters (same as Python)
params.dcomp = [2, 2, 2];
params.dsize = [16, 16, 16];

lpfilt = '9-7';


A = [
    1 2 3 4;
    4 5 6 7;
    7 8 9 10;
    10 11 12 13
];

K = [
    1 2;
    3 4
];

disp('Input A:');
disp(A);

disp('Kernel K:');
disp(K);

% FULL
full_mat = conv2(A, K, 'full');
disp('[MATLAB] FULL:');
disp(full_mat);

% SAME
same_mat = conv2(A, K, 'same');
disp('[MATLAB] SAME:');
disp(same_mat);

% VALID
valid_mat = conv2(A, K, 'valid');
disp('[MATLAB] VALID:');
disp(valid_mat);
% [dst, shear_f] = nsst_dec1e(img, params, lpfilt);

% stats(wind, "windowing output");
