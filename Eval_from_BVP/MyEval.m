function [ me, E_std, mae, rmse, mer, p ] = MyEval( HR_pr,HR_rel )
temp = HR_pr-HR_rel;
me = mean(temp);
E_std = std(temp);
mae = sum(abs(temp))/length(temp);
rmse = sqrt(sum(temp.*temp)/length(temp));
for i=1:length(temp)
    mer(i)=abs(temp(i))./(HR_rel(i)+0.01);
end
mer = mean(mer);
cc = corrcoef(HR_pr,HR_rel);
p = cc(1,2);

end
