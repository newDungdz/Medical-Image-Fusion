current = fileparts(mfilename('fullpath')); 
addpath(genpath(fullfile(current, 'src', 'matlab_metrics')));

clc;

% AArun_batch_metrics('data/AANLIB/MyDatasets/CT-MRI/test', 'data/Fused_results/CT-MRI', 'data/Evaluation_results/CT-MRI');
AArun_batch_metrics('data/AANLIB/MyDatasets/PET-MRI/test', 'data/Fused_results/PET-MRI', 'data/Evaluation_results/PET-MRI');
AArun_batch_metrics('data/AANLIB/MyDatasets/SPECT-MRI/test', 'data/Fused_results/SPECT-MRI', 'data/Evaluation_results/SPECT-MRI');
