clc
clear

STMap_Path1 = '.\STMap1';
STMap_Path2 = '.\STMap2';
STMap_list = dir(STMap_Path1);
STMap_list=STMap_list(~ismember({STMap_list.name},{'.','..'}));
z = 0;
for p=1:1:size(STMap_list)
    STMap1_path = fullfile(STMap_Path1, STMap_list(p).name);
    STMap1 =load(STMap1_path);
    STMap1 = (STMap1.STMap_pr)*128;
    
    STMap2_path = fullfile(STMap_Path2, STMap_list(p).name);
    STMap2 =load(STMap2_path);
    STMap2 = (STMap2.STMap_pr+1)*128;
    for j = size(STMap1,1)
        z = z + 1;
        STMap_temp1 = squeeze(STMap1(j,:,:,:));
        STMap_temp2 = squeeze(STMap2(j,:,:,:));
        STMap_temp1 = STMapN(STMap_temp1);
        STMap_temp2 = STMapN(STMap_temp2);
        MAE(z) = mean(mean(mean(abs(STMap_temp1-STMap_temp2))));
        peaksnr(z) = psnr(STMap_temp1,STMap_temp2,255);
        S1 = reshape(STMap_temp1,[], 1);
        S2 = reshape(STMap_temp2,[], 1);
        A=sqrt(sum(S1.^2));
        B=sqrt(sum(S2.^2));
        C=sum(sum(S1.*S2));
        YX(z)=C/(A*B);
        SSIM(z) = getMSSIM(STMap_temp1,STMap_temp2);
    end
end
mean(MAE)
mean(peaksnr)
mean(YX)
mean(SSIM)

