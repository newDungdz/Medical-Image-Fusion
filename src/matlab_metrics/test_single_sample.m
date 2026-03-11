clc
clear all
easy = 1;

% img1 = imread('test_sample/mri-ct/mri.png');
% img2 = imread('test_sample/mri-ct/ct.png');
% img_f = imread('test_sample/mri-ct/fused/fused_dwt.png');

img1 = imread('test_sample/ir-vi/ir.png');
img2 = imread('test_sample/ir-vi/vi.png');
img_f = imread('test_sample/ir-vi/fused/DenseFuse.png');

if size(img1, 3)>2
    img1 = rgb2gray(img1);
end

if size(img2, 3)>2
    img2 = rgb2gray(img2);
end

if size(img_f, 3)>2
    img_f = rgb2gray(img_f);
end

grey_level = 256;
[s1,s2] = size(img1);
img1_int = img1;
img2_int = img2;
img_f_int = img_f;
img1_float = im2double(img1) * 255.0;
img2_float = im2double(img2) * 255.0;
img_f_float = im2double(img_f) * 255.0;
imgSeq = zeros(s1, s2, 2);
imgSeq(:, :, 1) = img1_float;
imgSeq(:, :, 2) = img2_float;
fprintf('Size: %d x %d | Min: %d | Max: %d | Class: %s\n', s1, s2, min(img_f_float(:)), max(img_f_float(:)), class(img_f_float))
if easy ==1
    %EN
    EN = EN_metrics(img_f_int);        
    %MI
    MI = MI_metrics(img1_int, img2_int, img_f_int, grey_level);        
    %PSNR
    PSNR = PSNR_metrics(img1_float, img2_float, img_f_float);
    %MSE
    MSE = MSE_metrics(img1_float, img2_float, img_f_float);
    %SF
    SF = SF_metrics(img_f_float);
    %SD
    SD = SD_metrics(img_f_float);
    %VIF
    VIF = VIF_metrics(img1_float, img_f_float) + VIF_metrics(img2_float, img_f_float);
    %AG
    AG = AG_metrics(img_f_float);
    %CC
    CC = CC_metrics(img1_float, img2_float, img_f_float);
    %SCD
    SCD = SCD_metrics(img1_float, img2_float, img_f_float);        %Qabf
    Qabf = QABF_metrics(img1_float, img2_float, img_f_float);
    fprintf('EN: %f\n', EN);
    fprintf('MI: %f\n', MI);
    fprintf('PSNR: %f\n', PSNR);
    fprintf('MSE: %f\n', MSE);
    fprintf('SF: %f\n', SF);
    fprintf('SD: %f\n', SD);
    fprintf('VIF: %f\n', VIF);
    fprintf('AG: %f\n', AG);
    fprintf('CC: %f\n', CC);
    fprintf('SCD: %f\n', SCD);
    fprintf('Qabf: %f\n', Qabf);
else
    Nabf = NABF_metrics(img1_float, img2_float, img_f_float);
    % SSIM_a
    SSIM = SSIM_metrics(img1_float, img2_float, img_f_float);
    %MS_SSIM
    [MS_SSIM,t1,t2]= MS_SSIM_metrics(imgSeq, img_f_float);
    %FMI
    FMI_pixel = FMI_metrics(img1_float, img2_float, img_f_float);
    FMI_dct = FMI_metrics(img1_float, img2_float, img_f_float,'dct');
    FMI_w = FMI_metrics(img1_float, img2_float, img_f_float,'wavelet');
    FMI_edge = FMI_metrics(img1_float, img2_float, img_f_float,'edge');
    fprintf('NABF: %f\n', Nabf);
    fprintf('SSIM: %f\n', SSIM);
    fprintf('MS_SSIM: %f\n', MS_SSIM);
    fprintf('FMI_pixel: %f\n', FMI_pixel);
    fprintf('FMI_dct: %f\n', FMI_dct);
    fprintf('FMI_w: %f\n', FMI_w);
    fprintf('FMI_edge: %f\n', FMI_edge);
end

