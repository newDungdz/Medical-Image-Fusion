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
[dst, shear_f] = nsst_dec1e(img, params, lpfilt);

stats(img, '[original]');
stats(dst{1}, '[dst{1} - lowpass]');

count = 1;
for i = 2:length(dst)
    band = dst{i};
    for d = 1:size(band,3)
        stats(band(:,:,d), sprintf('[dst %d dir %d]', count, d));
    end
    count = count + 1;
end

% -----------------------------
% Reconstruction
% -----------------------------
rec = nsst_rec1(dst, lpfilt);

stats(rec, '[reconstructed]');

% -----------------------------
% Error
% -----------------------------
error = norm(img(:) - rec(:)) / norm(img(:));
fprintf('Relative reconstruction error: %.6e\n', error);