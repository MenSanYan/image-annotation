    
    resultFilse{1} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_01*'));
    options.labels{2} = 'tu=1 tis';
    resultFilse{2} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_2*'));
    options.labels{3} = 'tu=2 tis';
    resultFilse{3} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_3*'));
    options.labels{4} = 'tu=3 tis';
    resultFilse{4} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_4*'));
    options.labels{5} = 'tu=4 tis';
    resultFilse{5} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_5*'));
    options.labels{6} = 'tu=5 tis';
    resultFilse{6} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_6*'));
    options.labels{7} = 'tu=6 tis';
    resultFilse{7} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_7*'));
    options.labels{8} = 'tu=7 tis';
    resultFilse{8} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_8*'));
    options.labels{9} = 'tu=8 tis';
    resultFilse{9} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_9*'));
    options.labels{10} = 'tu=9 tis';
    resultFilse{10} = dir(fullfile('./','results_seperatelyNMF_1000_beta0.0001_L2_2_d300_tu_10*'));
    options.labels{11} = 'tu=10 tis';
    
    num = size(resultFilse,2);
    data = zeros(10,10,4);
    for k=1:num
        numofelement = size(resultFilse{k},1);
        for i=1:numofelement
            tmp = zeros(1,4);
            load(resultFilse{k}(i).name);
            times = size(results,1);
            for j=1:times
                [~,id] = max(results{j}(:,3));
                tmp = tmp + results{j}(id,:);
            end
            tmp = tmp/times;
            data(k,i,1) = tmp(1);
            data(k,i,2) = tmp(2);
            data(k,i,3) = tmp(3);
            data(k,i,4) = tmp(4);
        end
    end
    
    xx = [0.28,0.26,0.28,144];
    x_x = [0.01,0.01,0.005,2];
    xxx = [0.35,0.33,0.33,162];
    tis = [0:10];
    labels = {'pre','rec','f1','N+'};
    figure;
    for i=1:4
       subplot(2,2,i);
       tmp = data(:,:,i)-xx(i);
       tmp(tmp<0)=0;
       bar3(tmp);
%        set(gca,'ztick',[xx(i):x_x(i):xxx(i)]);
       set(gca,'zticklabel',[xx(i):x_x(i):xxx(i)]);
       set(gca,'xticklabel',tis);
       xlabel('tl');
       ylabel('tu');
       zlabel(labels{i});
    end