% this file draw the effect of args (alpha dimension) for multi-view nmf with Graph embedding
% algorithm.
    resultFilse1 = {};
    resultFilse2 = {};
    basePath = './';
    resultOri = dir(fullfile(basePath,'results_3feature.mat'));

%% dimension  for algorithm 
    resultFilse1{1} = dir(fullfile(basePath,'results_seperatelyNMF_alpha0_beta0_L2_2_d*'));
    options1.labels{1} = 'alpha=0';
    resultFilse1{2} = dir(fullfile(basePath,'results_seperatelyNMF_alpha1000_beta0.0001_L2_2_d*'));
    options1.labels{2} = 'alpha=1000';   
    resultFilse1{3} = dir(fullfile(basePath,'results_seperatelyNMF_alpha2000_beta0.0001_L2_2_d*'));
    options1.labels{3} = 'alpha=2000';

    options1.type = 'avg_best_F1';
    sub_caculateResults(resultOri,resultFilse1,options1);

%%alphas_K=300
    resultFilse2{1} = dir(fullfile(basePath,'results_seperatelyNMF_alpha*_beta*_L2_2_d300*'));
    d3000 = dir(fullfile(basePath,'results_seperatelyNMF_alpha3000_beta0.0001_L2_2_d300.mat'));
    resultFilse2{1}(end+1) = d3000;
    options2.labels{1} = 'test alpha';
    

    options2.type = 'avg_best_F1';
    sub_caculateResults(resultOri,resultFilse2,options2);