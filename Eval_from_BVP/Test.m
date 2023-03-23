clc
clear

save_Path = 'D:\Code\WAVE_TEST\Wave_Sort';
Wave_list = dir(save_Path);
Wave_list=Wave_list(~ismember({Wave_list.name},{'.','..'}));
fn = 256;
for p=1:2:size(Wave_list)
    now_gt = fullfile(save_Path, Wave_list(p).name);
    now_pr = fullfile(save_Path, Wave_list(p+1).name);
    Wave_gt = load(now_gt);
    Wave_gt = Wave_gt.Wave;
    Wave_pr = load(now_pr);
    Wave_pr = Wave_pr.Wave;
    num = size(Wave_gt, 1);
    Index_gt = zeros(num, 6);
    Index_pr = zeros(num, 6);
    for n = 1:num
        Wave_PR_One = Wave_pr(n,:);
        Wave_GT_One = Wave_gt(n,:);
        x = 1:fn;
        xx = 1:0.1169:fn;      % 30/256 = 0.1172
        Wave_PR_One=interp1(x,double(Wave_PR_One),xx,'spline');  
        Wave_GT_One=interp1(x,double(Wave_GT_One),xx,'spline');    
        [HRV_pr, ~, ~, HR1_pr, HR2_pr] = ExtractFeatures_test_nofilters_only_HRV(Wave_PR_One,'allrange');
        [HRV_gt, ~, ~, HR1_gt, HR2_gt] = ExtractFeatures_test_nofilters_only_HRV(Wave_GT_One,'allrange');
        Index_pr(n,:) = [HRV_pr(1,:), HR1_pr(1), HR2_pr(1)];
        Index_gt(n,:) = [HRV_gt(1,:), HR1_gt(1), HR2_gt(1)];
    end
    Index_ALL_pr{floor(p/2)+1} = Index_pr;
    Index_ALL_gt{floor(p/2)+1} = Index_gt;
    Index_ALL_pr_mean(floor(p/2)+1,:) = mean(Index_pr, 1);
    Index_ALL_gt_mean(floor(p/2)+1,:) = mean(Index_gt, 1);
end

for f = 1:6
    [me(f), E_std(f), mae(f), rmse(f), mer(f), p(f)] = MyEval(Index_ALL_pr_mean(:,f),Index_ALL_gt_mean(:,f));
end
result = [me; E_std; mae; rmse; mer; p]
