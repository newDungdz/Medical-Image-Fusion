clc; clear;

% --- Small synthetic image ---
I = double([
    10 10 10 10 10;
    10 50 50 50 10;
    10 50 100 50 10;
    10 50 50 50 10;
    10 10 10 10 10
]);

% --- Sobel filters (same as edge uses internally) ---
kx = [1 0 -1; 2 0 -2; 1 0 -1];
ky = [1 2 1; 0 0 0; -1 -2 -1];

% --- Gradients ---
bx = conv2(I, kx, 'same');
by = conv2(I, ky, 'same');

% --- Parameters ---
offset = 0;                 % usually 0 for grayscale
eps_val = 100 * eps;        % same as edge()
cutoff = 0.1;               % try small threshold

% --- Call internal function ---
[eout, thresh, gv, gh] = images.internal.builtins.computeEdges( ...
    I, bx, by, kx, ky, int8(offset), eps_val, cutoff);

% --- Display ---
disp('Input:'); disp(I);
disp('Gradient X (bx):'); disp(bx);
disp('Gradient Y (by):'); disp(by);

disp('Edge Output (eout):'); disp(eout);
disp('Threshold returned:'); disp(thresh);

figure;
subplot(1,2,1); imagesc(I); title('Input'); axis image; colormap gray;
subplot(1,2,2); imagesc(eout); title('computeEdges Output'); axis image;