current = fileparts(mfilename('fullpath'));
addpath(genpath(current))
clc;

% img1 = imread('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/3015.png');
% img2 = imread('data/AANLIB/MyDatasets/SPECT-MRI/test/SPECT/3015.png');
% img_f = imread('data/ASFE-Fusion-result/SPECT-MRI/3015.png');

% ========================
% TEST MATRIX 3x3
% ========================
img1 = uint8([80  20  85;
              75  25  78;
              80  22  88]);

img2 = uint8([30 110  35;
              28 120  32;
              26 115  30]);

img_f = uint8([58  70  60;
               55  78  57;
               62  75  61]);

% ========================
% TEST MATRIX 5x5
% ========================

% img1 = uint8([85  20  88  22  90;
%               80  25  82  24  84;
%               78  23  80  21  79;
%               82  26  85  25  87;
%               88  22  90  23  92]);

% img2 = uint8([30 110  32 115  35;
%               28 120  30 118  33;
%               26 115  28 112  31;
%               29 122  31 120  34;
%               27 118  29 116  32]);

% img_f = uint8([58  70  60  72  62;
%                55  78  57  76  59;
%                52  72  54  70  56;
%                56  80  58  78  60;
%                57  75  59  74  61]);

% ========================
% PREPROCESS
% ========================
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

% =========================================================
% 1. GLOBAL METRICS (NO sliding window) → ALWAYS SAFE
% =========================================================
fprintf('\n=== GLOBAL METRICS ===\n');

EN   = EN_metrics(img_f_int);
OCE  = OCE_metrics(img1_float, img2_float, img_f_float);
MI   = MI_metrics(img1_int, img2_int, img_f_int, grey_level, 0);
NMI   = MI_metrics(img1_int, img2_int, img_f_int, grey_level, 1);
[MLI_F, MLI_ref] = MLI_metrics(img1_float, img2_float, img_f_float);
PSNR = PSNR_metrics(img1_float, img2_float, img_f_float);
MSE  = MSE_metrics(img1_float, img2_float, img_f_float);
SF   = SF_metrics(img_f_float);
SD   = SD_metrics(img_f_float);
AG   = AG_metrics(img_f_float);
CC   = CC_metrics(img1_float, img2_float, img_f_float);
SCD  = SCD_metrics(img1_float, img2_float, img_f_float);
EI   = EI_metrics(img_f_float);
QTE   = QTE_metrics(img1_int, img2_int, img_f_int, 0.43137); % Nava constants
rSFe = rSFe_metrics(img1_float, img2_float, img_f_float);

fprintf('EN   = %f\n', EN);
fprintf('OCE  = %f\n', OCE);
fprintf('MI   = %f\n', MI);
fprintf('NMI   = %f\n', NMI);
fprintf('MLI_F = %f\n', MLI_F);
fprintf('MLI_ref = %f\n', MLI_ref);
fprintf('PSNR = %f\n', PSNR);
fprintf('MSE  = %f\n', MSE);
fprintf('SF   = %f\n', SF);
fprintf('SD   = %f\n', SD);
fprintf('AG   = %f\n', AG);
fprintf('CC   = %f\n', CC);
fprintf('SCD  = %f\n', SCD);
fprintf('EI   = %f\n', EI);
fprintf('QTE   = %f\n', QTE);
fprintf('rSFe = %f\n', rSFe);


% =========================================================
% 2. SMALL-NEIGHBOR METRICS (edge/gradient-based)
% → vẫn chạy được với 3x3
% =========================================================
fprintf('\n=== SMALL-NEIGHBOR METRICS ===\n');

try
    
    FMI_pixel = FMI_metrics(img1_float, img2_float, img_f_float);
    FMI_dct   = FMI_metrics(img1_float, img2_float, img_f_float, 'dct');
    FMI_w     = FMI_metrics(img1_float, img2_float, img_f_float, 'wavelet');
    FMI_edge  = FMI_metrics(img1_float, img2_float, img_f_float, 'edge');

    Qabf_old = QABF_metrics(img1_float, img2_float, img_f_float);
    [QABF, LABF, Nabf_K, Nabf] = Petrovic_metrics(img_f_float, img1_float, img2_float);
    NCIE = NCIE_metrics(img1_float, img2_float, img_f_float);

    SSIM = SSIM_metrics(img1_float, img2_float, img_f_float);
    [MEF_SSIM, ~, ~] = MEF_SSIM_metrics(imgSeq, img_f_float);

    VIF  = VIF_metrics(img1_float, img_f_float) + ...
        VIF_metrics(img2_float, img_f_float);

    VIFF = VIFF_metrics(img1_float, img2_float, img_f_float);

    fprintf('FMI_pixel = %f\n', FMI_pixel);
    fprintf('FMI_dct   = %f\n', FMI_dct);
    fprintf('FMI_w     = %f\n', FMI_w);
    fprintf('FMI_edge  = %f\n', FMI_edge);
    fprintf('Qabf_old = %f\n', Qabf_old);
    fprintf('QABF = %f\n', QABF);
    fprintf('LABF = %f\n', LABF);
    fprintf('NABF = %f\n', Nabf);
    fprintf('NABF_K = %f\n', Nabf_K);
    fprintf('NCIE = %f\n', NCIE);
    fprintf('\nSSIM = %f\n', SSIM);
    fprintf('MEF_SSIM = %f\n', MEF_SSIM);
    fprintf('VIF  = %f\n', VIF);
    fprintf('VIFF = %f\n', VIFF);
catch
    fprintf('[WARNING] Small-neighbor metrics failed (border issue)\n');
end


% =========================================================
% 3. TRUE SLIDING WINDOW METRICS (SSIM-like)
% → REQUIRE LARGE IMAGE (>= 11x11)
% =========================================================
fprintf('\n=== SLIDING WINDOW METRICS ===\n');



Qcv  = QCV_metrics(img1_float, img2_float, img_f_float);
Qcb  = QCB_metrics(img1_float, img2_float, img_f_float);
QP   = QP_metrics(img1_float, img2_float, img_f_float);
% Qw   = Peilla_metrics(img1_float, img2_float, img_f_float, 1);
% Qe   = Peilla_metrics(img1_float, img2_float, img_f_float, 2);
% QC   = QC_metrics(img1_float, img2_float, img_f_float, 2);
% QY   = QY_metrics(img1_float, img2_float, img_f_float);


fprintf('Qcv  = %f\n', Qcv);
fprintf('Qcb  = %f\n', Qcb);
% fprintf('Qw   = %f\n', Qw);
% fprintf('Qe   = %f\n', Qe);
% fprintf('QC   = %f\n', QC);
fprintf('QP   = %f\n', QP);
% fprintf('QY   = %f\n', QY);