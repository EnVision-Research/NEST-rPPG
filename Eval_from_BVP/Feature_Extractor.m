function Feature_Vec=Feature_Extractor(NNI)
    Feature_Vec = [];
    
    [pxx,f] = plomb(NNI,cumsum(NNI)./256);

    % obtain RF with the peak of (f>0.15 & f<0.4)
    ff = [];
    for nn=1:length(f)
        if f(nn)>0.15 && f(nn)<0.4
            ff=[ff,f(nn)];
        end
    end
    [peaks, locs]=findpeaks(pxx(f>0.15 & f<0.4),'NPeaks', 1);
    if length(locs)<1   % if there is no peak
        [peaks, locs]=max(pxx(f>0.15 & f<0.4));
        disp('no peak, find max instead RF')
        disp(locs)
    end
    RF = ff(locs);
    
    LF=sum(pxx(f>0.04 & f<0.15));
    HF=sum(pxx(f>0.15 & f<0.4));
    LF_un=LF/(LF+HF);
    HF_un=HF/(LF+HF);
    ratio_F=LF_un/HF_un;
    Feature_Vec=[Feature_Vec LF_un];
    Feature_Vec=[Feature_Vec HF_un];
    Feature_Vec=[Feature_Vec ratio_F];
    Feature_Vec=[Feature_Vec RF];
return




