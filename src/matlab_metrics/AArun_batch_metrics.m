current = fileparts(mfilename('src/matlab_metrics'));
addpath(genpath(current))

clc;

easy_metric = true;
hard_metric = false;

run_evaluation('data/AANLIB/MyDatasets/CT-MRI/test', 'data/Fused_results/CT-MRI', 'data/Evaluation_results/CT-MRI', easy_metric, hard_metric);
% run_evaluation('data/AANLIB/MyDatasets/PET-MRI/test', 'data/Fused_results/PET-MRI', 'data/Evaluation_results/PET-MRI', easy_metric, hard_metric);
% run_evaluation('data/AANLIB/MyDatasets/SPECT-MRI/test', 'data/Fused_results/SPECT-MRI', 'data/Evaluation_results/SPECT-MRI', easy_metric, hard_metric);


function run_evaluation(root_folder, fused_root, output_folder, easy_metric, hard_metric)

grey_level = 256;

% parameter for Tsallis entropy (TE) metric
te_q = 1.5; % q != 1

% ── Discover source image folders ─────────────────────────────────────────────
% root_folder contains exactly 2 subfolders (the two source modalities)
src_entries = dir(root_folder);
src_entries = src_entries([src_entries.isdir] & ~startsWith({src_entries.name}, '.'));

assert(numel(src_entries) == 2, ...
    'Expected exactly 2 source folders in "%s", found %d.', root_folder, numel(src_entries));

% Build full paths so imread can locate files
src_dirs = { fullfile(src_entries(1).folder, src_entries(1).name), ...
             fullfile(src_entries(2).folder, src_entries(2).name) };

% ── Discover method subfolders inside fused root ───────────────────────────────
fused_entries = dir(fused_root);
fused_entries = fused_entries([fused_entries.isdir] & ~startsWith({fused_entries.name}, '.'));

assert(~isempty(fused_entries), 'No method folders found inside "%s".', fused_root);

fprintf('Source 1   : %s\n', src_dirs{1});
fprintf('Source 2   : %s\n', src_dirs{2});
fprintf('Fused root : %s\n', fused_root);
for f = 1:numel(fused_entries)
    fprintf('  Method %d : %s\n', f, fused_entries(f).name);
end

% ── Output directory ──────────────────────────────────────────────────────────
out_dir = fullfile(output_folder);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ── Supported image extensions ────────────────────────────────────────────────
IMG_EXT = {'*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff'};

% ── Column definitions ───────────────────────────────────────────────────────
% NOTE: Some metrics are computed even when not all are written out below. Ensure
% these match the values stored in the per-image/result tables.
easy_cols         = {'ImageName','EN','MI','PSNR','MSE','SF','SD','VIF','AG','CC','EI','NCIE','OCE','SCD','Qabf','Qcv','Qcb','Qw','Qe','QC1','QC2','QP','QY','TE','rSFe','VIFF'};
hard_cols         = {'ImageName','NABF','SSIM','MEF_SSIM','FMI_pixel','FMI_dct','FMI_wavelet','FMI_edge'};
summary_easy_cols = {'Model',    'EN','MI','PSNR','MSE','SF','SD','VIF','AG','CC','EI','NCIE','OCE','SCD','Qabf','Qcv','Qcb','Qw','Qe','QC1','QC2','QP','QY','TE','rSFe','VIFF'};
summary_hard_cols = {'Model',    'NABF','SSIM','MEF_SSIM','FMI_pixel','FMI_dct','FMI_wavelet','FMI_edge'};

% ── Summary accumulators (one row per method) ─────────────────────────────────
summary_easy = {};
summary_hard = {};

% ── Process each method folder ────────────────────────────────────────────────
for fi = 1:numel(fused_entries)
    method_name = fused_entries(fi).name;
    fused_dir   = fullfile(fused_root, method_name);

    fprintf('\n==============================\n');
    fprintf('Processing: %s\n', method_name);
    fprintf('==============================\n');

    % Collect image files from this method folder
    img_files = [];
    for e = 1:numel(IMG_EXT)
        found = dir(fullfile(fused_dir, IMG_EXT{e}));
        img_files = [img_files; found]; %#ok<AGROW>
    end

    if isempty(img_files)
        warning('No images found in %s, skipping.', fused_dir);
        continue;
    end

    easy_data = {};
    hard_data = {};

    for ii = 1:numel(img_files)
        img_name = img_files(ii).name;
        fprintf('[%d/%d] %s ... ', ii, numel(img_files), img_name);

        % ── Load images ───────────────────────────────────────────────────
        path1 = fullfile(src_dirs{1}, img_name);
        path2 = fullfile(src_dirs{2}, img_name);
        pathF = fullfile(fused_dir,   img_name);

        if ~isfile(path1) || ~isfile(path2)
            fprintf('SKIPPED (source image not found)\n');
            continue;
        end

        img1  = ensureGray(imread(path1));
        img2  = ensureGray(imread(path2));
        img_f = ensureGray(imread(pathF));

        [s1, s2] = size(img1);

        img1_int  = img1;
        img2_int  = img2;
        img_f_int = img_f;

        img1_float  = im2double(img1) * 255.0;
        img2_float  = im2double(img2) * 255.0;
        img_f_float = im2double(img_f) * 255.0;

        imgSeq = zeros(s1, s2, 2);
        imgSeq(:,:,1) = img1_float;
        imgSeq(:,:,2) = img2_float;

        % ── Easy metrics ──────────────────────────────────────────────────
        if easy_metric
            try
                AG   = AG_metrics(img_f_float);
                CC   = CC_metrics(img1_float, img2_float, img_f_float);
                EI   = EI_metrics(img_f_float);
                EN   = EN_metrics(img_f_int);
                MI   = MI_metrics(img1_int, img2_int, img_f_int, grey_level);
                MSE  = MSE_metrics(img1_float, img2_float, img_f_float);
                NCIE = NCIE_metrics(img1_float, img2_float, img_f_float);
                OCE  = OCE_metrics(img1_float, img2_float, img_f_float);
                PSNR = PSNR_metrics(img1_float, img2_float, img_f_float);
                SF   = SF_metrics(img_f_float);
                SD   = SD_metrics(img_f_float);
                VIF  = VIF_metrics(img1_float, img_f_float) + VIF_metrics(img2_float, img_f_float);
                SCD  = SCD_metrics(img1_float, img2_float, img_f_float);
                Qabf = QABF_metrics(img1_float, img2_float, img_f_float);
                Qcv   = QCV_metrics(img1_float, img2_float, img_f_float);
                Qcb   = QCB_metrics(img1_float, img2_float, img_f_float);
                Qw = Peilla_metrics(img1_float, img2_float, img_f_float, 1);
                Qe = Peilla_metrics(img1_float, img2_float, img_f_float, 2);

                % Additional metrics (not previously included in output)
                QC1  = QC_metrics(img1_float, img2_float, img_f_float, 1);
                QC2  = QC_metrics(img1_float, img2_float, img_f_float, 2);
                QP   = QP_metrics(img1_float, img2_float, img_f_float);
                QY   = QY_metrics(img1_float, img2_float, img_f_float);
                TE   = TE_metrics(img1_int, img2_int, te_q);
                rSFe = rSFe_metrics(img1_float, img2_float, img_f_float);
                VIFF = VIFF_metrics(img1_float, img2_float, img_f_float);

                easy_data(end+1,:) = {img_name, EN, MI, PSNR, MSE, SF, SD, VIF, AG, CC, EI, NCIE, OCE, SCD, Qabf, Qcv, Qcb, Qw, Qe, QC1, QC2, QP, QY, TE, rSFe, VIFF}; %#ok<AGROW>
            catch e_easy
                warning('\nEasy metrics failed for %s: %s', img_name, e_easy.message);
            end
        end

        % ── Hard metrics ──────────────────────────────────────────────────
        if hard_metric
            try
                [LABF, NABF]      = Petrovic_metrics(img1_float, img2_float, img_f_float);
                SSIM      = SSIM_metrics(img1_float, img2_float, img_f_float);
                [MEF_SSIM, ~, ~] = MEF_SSIM_metrics(imgSeq, img_f_float);
                FMI_pixel = FMI_metrics(img1_float, img2_float, img_f_float);
                FMI_dct   = FMI_metrics(img1_float, img2_float, img_f_float, 'dct');
                FMI_w     = FMI_metrics(img1_float, img2_float, img_f_float, 'wavelet');
                FMI_edge  = FMI_metrics(img1_float, img2_float, img_f_float, 'edge');

                hard_data(end+1,:) = {img_name, LABF, NABF, SSIM, MEF_SSIM, FMI_pixel, FMI_dct, FMI_w, FMI_edge}; %#ok<AGROW>
            catch e_hard
                warning('\nHard metrics failed for %s: %s', img_name, e_hard.message);
            end
        end

        fprintf('done\n');
    end % image loop

    % ── Build and save per-method easy table ──────────────────────────────
    if easy_metric && ~isempty(easy_data)
        T_easy   = cell2table(easy_data, 'VariableNames', easy_cols);
        avg_easy = computeAverageRow(T_easy, easy_cols, 'AVERAGE');

        easy_csv = fullfile(out_dir, sprintf('%s_easy_metrics.csv', method_name));
        writetable([T_easy; avg_easy], easy_csv);
        fprintf('Easy metrics saved to: %s\n', easy_csv);

        % Accumulate into summary
        sum_easy_row           = avg_easy;
        sum_easy_row.ImageName = {method_name};
        summary_easy(end+1,:)  = table2cell(sum_easy_row); %#ok<AGROW>
    end

    % ── Build and save per-method hard table ──────────────────────────────
    if hard_metric && ~isempty(hard_data)
        T_hard   = cell2table(hard_data, 'VariableNames', hard_cols);
        avg_hard = computeAverageRow(T_hard, hard_cols, 'AVERAGE');

        hard_csv = fullfile(out_dir, sprintf('%s_hard_metrics.csv', method_name));
        writetable([T_hard; avg_hard], hard_csv);
        fprintf('Hard metrics saved to: %s\n', hard_csv);

        % Accumulate into summary
        sum_hard_row           = avg_hard;
        sum_hard_row.ImageName = {method_name};
        summary_hard(end+1,:)  = table2cell(sum_hard_row); %#ok<AGROW>
    end

end % method loop

% ── Write summary CSVs (only when multiple methods exist) ─────────────────────
if numel(fused_entries) > 1
    if easy_metric && ~isempty(summary_easy)
        T_sum_easy = cell2table(summary_easy, 'VariableNames', summary_easy_cols);
        writetable(T_sum_easy, fullfile(out_dir, 'summary_easy_metrics.csv'));
        fprintf('\nSummary (easy) saved to: %s\n', fullfile(out_dir, 'summary_easy_metrics.csv'));
    end

    if hard_metric && ~isempty(summary_hard)
        T_sum_hard = cell2table(summary_hard, 'VariableNames', summary_hard_cols);
        writetable(T_sum_hard, fullfile(out_dir, 'summary_hard_metrics.csv'));
        fprintf('Summary (hard) saved to: %s\n', fullfile(out_dir, 'summary_hard_metrics.csv'));
    end
end

fprintf('\nAll done. Results in: %s\n', out_dir);

end % run_evaluation


% ═════════════════════════════════════════════════════════════════════════════
%  Helper functions
% ═════════════════════════════════════════════════════════════════════════════

function img = ensureGray(img)
    if size(img, 3) > 2
        img = rgb2gray(img);
    end
end

function avg_row = computeAverageRow(T, col_names, label)
    avg_cell    = cell(1, numel(col_names));
    avg_cell{1} = label;
    for c = 2:numel(col_names)
        col_data = T.(col_names{c});
        if isnumeric(col_data)
            avg_cell{c} = mean(col_data, 'omitnan');
        else
            avg_cell{c} = NaN;
        end
    end
    avg_row = cell2table(avg_cell, 'VariableNames', col_names);
end