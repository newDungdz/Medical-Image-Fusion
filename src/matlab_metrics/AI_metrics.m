function [MI_F, MI_ref] = AI_metrics(A, B, F)
% AverageIntensityMetrics: Tính Average Intensity cho ảnh dung hợp và ảnh nguồn
%
% INPUT:
%   A - ảnh nguồn 1 (grayscale hoặc uint8/double)
%   B - ảnh nguồn 2 (grayscale hoặc uint8/double)
%   F - ảnh dung hợp
%
% OUTPUT:
%   MI_results - struct chứa các giá trị trung bình:
%                MI_A, MI_B, MI_F, MI_ref

    % Chuyển đổi ảnh về double để tính toán chính xác
    A = double(A);
    B = double(B);
    F = double(F);
    
    % Kích thước ảnh
    [M, N] = size(F);
    
    % Tính Mean Intensity cho từng ảnh
    MI_A = mean(A(:));
    MI_B = mean(B(:));
    MI_F = mean(F(:));
    
    % Mean Intensity tham chiếu theo công thức
    MI_ref = (MI_A + MI_B) / 2;
    
end