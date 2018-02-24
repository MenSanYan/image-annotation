function [avg_best_pre, std_best_pre, avg_best_rec, std_best_rec, avg_best_F1, std_best_F1, avg_best_Fp, std_best_Fp] = caculateScore(results)%,ts,avg_ts
    num = size(results,1);
    best_pre = zeros(num,4);
    best_rec = zeros(num,4);
    best_F1 = zeros(num,4);
    best_Fp = zeros(num,4);
    for i=1:num
        [~,id] = max(results{i}(:,1));
        best_pre(i,:) = results{i}(id,:);
        [~,id] = max(results{i}(:,2));
        best_rec(i,:) = results{i}(id,:);
        [~,id] = max(results{i}(:,3));
        best_F1(i,:) = results{i}(id,:);
        [~,id] = max(results{i}(:,4));
        best_Fp(i,:) = results{i}(id,:);
    end
    avg_best_pre = sum(best_pre)/num;
    std_best_pre = std(best_pre);
    
    avg_best_rec = sum(best_rec)/num;
    std_best_rec = std(best_rec);
    
    avg_best_F1 = sum(best_F1)/num;
    std_best_F1 = std(best_F1);
    
    avg_best_Fp = sum(best_Fp)/num;
    std_best_Fp = std(best_Fp);
    
%     avg_ts = sum(ts)/num;
end