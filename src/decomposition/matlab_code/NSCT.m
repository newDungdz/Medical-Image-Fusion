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

figure;

num_scales = length(y) - 1;
max_dirs = max(cellfun(@length, y(2:end)));

% Total rows = original + lowpass + scales
total_rows = num_scales + 2;

plot_idx = 1;

% --- Original ---
subplot(total_rows, max_dirs, plot_idx);
imshow(img, []);
title('Original');
plot_idx = plot_idx + 1;

% --- Lowpass ---
subplot(total_rows, max_dirs, plot_idx);
imshow(mat2gray(y{1}));
title('Lowpass');
plot_idx = plot_idx + max_dirs - 1;

% --- Directional subbands ---
for s = 2:length(y)
    subbands = y{s};
    
    for d = 1:length(subbands)
        subplot(total_rows, max_dirs, plot_idx);
        imshow(mat2gray(subbands{d}));
        title(sprintf('S%d-D%d', s-1, d));
        
        plot_idx = plot_idx + 1;
    end
    
    % move to next row start
    plot_idx = (s)*max_dirs + 1;
end

disp('NSCT decomposition done');