% Reference from https://github.com/Linfeng-Tang/Image-Fusion General Metrics

function mutural_informationR=MI_metrics(grey_matrixA,grey_matrixB,grey_matrixF,grey_level)
% mutural_informationR=mutural_information(grey_matrixA,grey_matrixB,grey_matrixF,grey_level)
% compute mutural information of the image
% grey_matrixA , grey_matrixB,grey_matrixF are grey values of imageA,imageB and fusion image
% grey_level is the grayscale degree of image
% please set grey_level=256
% ---------
% Author:  Qu Xiao-Bo    <quxiaobo [at] xmu.edu.cn>    June 26, 2009
%          Postal address:
% Rom 509, Scientific Research Building # 2,Haiyun Campus, Xiamen University,Xiamen,Fujian, P. R. China, 361005
% Website: http://quxiaobo.go.8866.org
HA=entropy(grey_matrixA);
HB=entropy(grey_matrixB);
HF=entropy(grey_matrixF);
HFA=Hab(grey_matrixF,grey_matrixA,grey_level);
HFB=Hab(grey_matrixF,grey_matrixB,grey_level);
MIFA=HA+HF-HFA;
MIFB=HB+HF-HFB;
mutural_informationR=MIFA+MIFB;

function HabR=Hab(grey_matrixA,grey_matrixB,grey_level)
% HabR=Hab(grey_matrixA,grey_matrixB,grey_level)
% compute mutural information of the image
% grey_matrixA , grey_matrixB,grey_matrixF are grey values of imageA,imageB and fusion image
% grey_level is the grayscale degree of image
% ---------
% Author:  Qu Xiao-bo    <quxiaobo429@163.com>    May 7,2006
%          Postal address:
%          Xiamen University, Department of Communication Engineering
%          Xiamen, Fujian, P. R. China, 361005  
[row,column]=size(grey_matrixA);
counter = zeros(256,256);
%ͳ��ֱ��ͼ
grey_matrixA=grey_matrixA+1;
grey_matrixB=grey_matrixB+1;
for i=1:row
    for j=1:column
        indexx = grey_matrixA(i,j);
        indexy = grey_matrixB(i,j);
        counter(indexx,indexy) = counter(indexx,indexy)+1;%����ֱ��ͼ
    end
end
%����������Ϣ��
total= sum(counter(:));
index = find(counter~=0);
p = counter/total;
HabR = sum(sum(-p(index).*log2(p(index))));
        
        
