% -------------------------------
% Contourlet Transform Decomposition
% -------------------------------

clc; clear;
current = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(current, 'contourlet_toolbox')));
% Read image
img = imread('test.png');
if size(img, 3) == 3
    img = rgb2gray(img);
end
img = im2double(img);
% Parameters
pfilt = '9-7';      % Pyramid filter
dfilt = 'pkva';     % Directional filter
nlevs = [0 2 3];    % Number of directional subbands at each scale

% Decomposition
y = pdfbdec(img, pfilt, dfilt, nlevs);

% Structure:
% y{1} = lowpass
% y{2}, y{3}, ... = directional subbands (cell arrays)
figure;
plot_idx = 1;

for l = 2:length(y)
    subbands = y{l};
    n_dir = length(subbands);
    
    for d = 1:n_dir
        subplot(length(y)-1, max(cellfun(@length, y(2:end))), plot_idx);
        imshow(subbands{d}, []);
        title(sprintf('Scale %d - Dir %d', l-1, d));
        
        plot_idx = plot_idx + 1;
    end
end

disp('CT decomposition done');