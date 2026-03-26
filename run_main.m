% run_main.m  – entry point for CI
current = fileparts(mfilename('fullpath'));   % fix: 'fullpath' not a custom path
addpath(genpath(fullfile(current, 'src', 'matlab_metrics')));

clc;

% run_evaluation('data/AANLIB/MyDatasets/CT-MRI/test',   'data/Fused_results/CT-MRI',   'data/Evaluation_results/CT-MRI');
run_evaluation('data/AANLIB/MyDatasets/PET-MRI/test',  'data/Fused_results/PET-MRI',  'data/Evaluation_results/PET-MRI');
run_evaluation('data/AANLIB/MyDatasets/SPECT-MRI/test','data/Fused_results/SPECT-MRI','data/Evaluation_results/SPECT-MRI');