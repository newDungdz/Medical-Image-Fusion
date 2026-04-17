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

lpfilt = 'maxflat';


[dst, shear_f] = nsst_dec1e(img, params, lpfilt);

% stats(wind, "windowing output");
