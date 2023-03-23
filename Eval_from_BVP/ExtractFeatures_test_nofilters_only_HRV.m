function [new_HRV,Rpeak,filtECG, HR_bpm, HR_psd_bpm]=ExtractFeatures_test_nofilters_only_HRV(signal,threshold_model)

fname = '1';
optargs = {0 30 0 'default'};  % default values for input arguments
[useSegments, windowSize, percentageOverlap] = optargs{:};
% clear optargs newVals
new_HRV = [];
% Parameters
fs = 256;       % sampling frequency [Hz]
% Add subfunctions to matlab path
addpath(genpath('./utils')) % add subfunctions folder to path
%% Run through files
if size(signal,1)<size(signal,2), signal = signal'; end % make sure it's column vector
signalraw =  signal;
% Figuring out if segmentation is used
if useSegments==1
    WINSIZE = windowSize; % window size (in sec)
    OLAP = percentageOverlap;
else
    WINSIZE = length(signal)/fs;
    OLAP=0;
end
startp = 1;
endp = WINSIZE*fs;
looptrue = true;
nseg = 1;
while looptrue
    % Conditions to stop loop
    if length(signal) < WINSIZE*fs
        endp = length(signal);
        looptrue = false;
        continue
    end
    if nseg > 1
        startp(nseg) = startp(nseg-1) + round((1-OLAP)*WINSIZE*fs);
        if length(signal) - endp(nseg-1) < WINSIZE*fs
            endp(nseg) = length(signal);
        else
            endp(nseg) = startp(nseg) + WINSIZE*fs -1;
        end
    end
    if endp(nseg) == length(signal)
        looptrue = false;
        nseg = nseg - 1;
    end
    nseg = nseg + 1;
end
if endp(nseg)-startp(nseg) < WINSIZE*fs-1
    nseg=nseg-1;
    endp(end)=[];
    startp(end)=[];
end


% Obtain features for each available segment
HR_bpm = [];
HR_psd_bpm = [];
filtECG = zeros(WINSIZE*fs,nseg); % save preprocessed ecg segments, nseg columns of signals.
Rpeak = cell(1,nseg+1); % save detected R peaks for each ecg segments.
for n = 1:nseg
    % Get signal of interest
    sig_seg = signal(startp(n):endp(n));
    % smooth using 64filter
    sig_seg_filtered = bpfilter64(sig_seg,fs);
    %sig_seg_filtered=detrend2(sig_seg_filtered,50);
    sig_seg_filtered = (sig_seg_filtered-mean(sig_seg_filtered))/std(sig_seg_filtered);
    filtECG(:,n) = sig_seg_filtered;     %sig_seg
    if strcmp(threshold_model,'default')
        [pks1,locs1]=findpeaks(sig_seg_filtered,'minpeakheight',0.01,'minpeakdistance',110);
        threshold = median(pks1)/4;
        [pks,locs]=findpeaks(sig_seg_filtered,'minpeakheight',threshold,'minpeakdistance',110);
    end
    if strcmp(threshold_model,'allrange')
        [pks,locs]=findpeaks(sig_seg_filtered,'minpeakdistance',110);
    end
    if strcmp(threshold_model,'zero_point_five')
        [pks,locs]=findpeaks(sig_seg_filtered,'minpeakheight',0.5,'minpeakdistance',110);
    end
    if strcmp(threshold_model,'relative_high')
        [pks1,locs1]=findpeaks(sig_seg_filtered,'minpeakheight',0.01,'minpeakdistance',110);
        threshold = median(pks1)/2;
        [pks,locs]=findpeaks(sig_seg_filtered,'minpeakheight',threshold,'minpeakdistance',110);
    end
    % QRS detect
    [qrsseg,featqrs] = multi_qrsdetect_test(sig_seg_filtered,fs,[fname '_s' num2str(n)]);
    qrsseg{end} = locs;
    Rpeak{1,n} = qrsseg{end};
    % HRV features in Iman new metric 
    % % MEAN_IBI,STD_IBI,RMSSD,RMSM,SD1,SD2,pNNI50,LF_n,HF_n,ratio_F
    Feature_Vec=Feature_Extractor(diff(qrsseg{end}));
    new_HRV = [new_HRV; Feature_Vec];
    % Heart Rate features
    %HRbpm = median(60./(diff(qrsseg{end}./fs)));
    HRbpm = mean(60./(diff(qrsseg{end}./fs)));
    HR_bpm = [HR_bpm, HRbpm];
    % PSD method for HR
    [Pg,f] = pwelch(sig_seg_filtered,[],[],2^13,fs);
    Frange = find(f>0.7&f<3); % consider the frequency within [0.7Hz, 4Hz].
    idxG = Pg == max(Pg(Frange));
    HR = f(idxG)*60;
    HR_psd_bpm = [HR_psd_bpm, HR];
end



