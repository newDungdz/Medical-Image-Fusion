clc; clear;

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

% -----------------------------
% Stats function (inline)
% -----------------------------
stats = @(x, name) fprintf([ ...
    '%s shape=(%d,%d) min=%.4f max=%.4f mean=%.4f sum=%.4f\n'], ...
    name, size(x,1), size(x,2), ...
    min(x(:)), max(x(:)), mean(x(:)), sum(x(:).^2));

% -----------------------------
% Decomposition
% -----------------------------
% -----------------------------
% Decomposition (timed)
% -----------------------------
tic;
[dst, shear_f] = nsst_dec1e(img, params, lpfilt);
t_dec = toc;
fprintf('Decomposition time: %.6f seconds\n', t_dec);

% -----------------------------
% Reconstruction (timed)
% -----------------------------
tic;
rec = nsst_rec1(dst, lpfilt);
t_rec = toc;
fprintf('Reconstruction time: %.6f seconds\n', t_rec);
psnr = 10 * log10(255^2 / mean((img(:) - rec(:)).^2));
fprintf('PSNR between original and reconstructed image: %.2f dB\n', psnr);