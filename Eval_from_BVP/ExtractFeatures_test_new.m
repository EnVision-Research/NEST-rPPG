function [HRV_feature,Rpeak,filtECG, HR_bpm]=ExtractFeatures_test_new(signal)

optargs = {1 30 0};  % default values for input arguments
[useSegments, windowSize, percentageOverlap] = optargs{:};
fname = '1';


fs = 256;       % sampling frequency [Hz]

% Add subfunctions to matlab path
addpath(genpath('./utils')) % add subfunctions folder to path


%% Initialize loop
% Wide BP
Fhigh = 5;  % highpass frequency [Hz]
Flow = 45;   % low pass frequency [Hz]
Nbut = 10;     % order of Butterworth filter
d_bp= design(fdesign.bandpass('N,F3dB1,F3dB2',Nbut,Fhigh,Flow,fs),'butter');
[b_bp,a_bp] = tf(d_bp);
HR_bpm = [];

% Narrow BP
clear Fhigh Flow Nbut d_bp

%% Run through files
if size(signal,1)<size(signal,2), signal = signal'; end % make sure it's column vector

%% Preprocessing
signal = filtfilt(b_bp,a_bp,signal);             % filtering narrow
signal = detrend(signal);                        % detrending (optional)
signal = signal - mean(signal);
signal = signal/std(signal);                     % standardizing

disp('Preprocessed...')

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
Rpeak = cell(1,nseg+1);
HRV_feature = [];
filtECG = zeros(WINSIZE*fs,nseg); 
for n = 1:nseg
    % Get signal of interest
    sig_seg = signal(startp(n):endp(n));
    filtECG(:,n) = sig_seg;

    % QRS detect
    [qrsseg,featqrs] = multi_qrsdetect_test(sig_seg,fs,[fname '_s' num2str(n)]);
    Rpeak{1,n} = qrsseg{end};
    
    HRbpm = mean(60./(diff(qrsseg{end}./fs)));
    HR_bpm = [HR_bpm, HRbpm];
    
    % calculate    LF, HF , LF_n, HF_n, ratio_F, RF
    Feature_Vec=Feature_Extractor(diff(qrsseg{end}));
    HRV_feature = [HRV_feature; Feature_Vec];

end

%% get result for the whole video
sig_seg = signal;

% QRS detect
[qrsseg,featqrs] = multi_qrsdetect_test(sig_seg,fs,[fname '_s' num2str(nseg+1)]);
Rpeak{1,nseg+1} = qrsseg{end};

HRbpm = mean(60./(diff(qrsseg{end}./fs)));
HR_bpm = [HR_bpm, HRbpm];

% calculate    LF, HF , LF_n, HF_n, ratio_F, RF
Feature_Vec=Feature_Extractor(diff(qrsseg{end}));
HRV_feature = [HRV_feature; Feature_Vec];


    

