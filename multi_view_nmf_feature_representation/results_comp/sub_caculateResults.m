function sub_caculateResults(oriFile,files,option)
%%
%args
%files is the results file, which is a struct.
%option.type :
%option.type :
%%  
    
    if ~isfield(option,'type')
        option.type = '';
    end
    
    numOftype = size(files,2);
    
    load(oriFile.name);
    [~,id] = max(perf(:,3));
    avg_best_pre = perf(id,:);
    avg_best_rec = perf(id,:);
    avg_best_F1 = perf(id,:);
    avg_best_Fp = perf(id,:);


    
    avg_best_pres = cell(numOftype,1);
    avg_best_recs = cell(numOftype,1);
    avg_best_F1s = cell(numOftype,1);
    avg_best_Fps = cell(numOftype,1);
    std_best_pres = cell(numOftype,1);
    std_best_recs = cell(numOftype,1);
    std_best_F1s = cell(numOftype,1);
    std_best_Fps = cell(numOftype,1);
    
for j=1:numOftype
    n = size(files{j},1);
    
    for i=1:n
        load(files{j}(i).name);
        [avg_best_pres{j}(i,:), std_best_pres{j}(i,:), avg_best_recs{j}(i,:), std_best_recs{j}(i,:), avg_best_F1s{j}(i,:), std_best_F1s{j}(i,:), avg_best_Fps{j}(i,:), std_best_Fps{j}(i,:)] = caculateScore(results);
    end

end

options.labels = option.labels;

switch option.type
    case 'avg_best_pre'
        draw_result_v2(9,avg_best_pre,avg_best_pres,std_best_pres,'avg_best_pre',options);
    case 'avg_best_rec'
        draw_result_v2(9,avg_best_rec,avg_best_recs,std_best_recs,'avg_best_rec',options);
    case 'avg_best_F1'
        draw_result_v2(9,avg_best_F1,avg_best_F1s,std_best_F1s,'avg_best_F1',options);
    case 'avg_best_Fp'
        draw_result_v2(9,avg_best_Fp,avg_best_Fps,std_best_Fps,'avg_best_Fp',options);
    otherwise 
        draw_result_v2(9,avg_best_pre,avg_best_pres,std_best_pres,'avg_best_pre',options);
        draw_result_v2(9,avg_best_rec,avg_best_recs,std_best_recs,'avg_best_rec',options);
        draw_result_v2(9,avg_best_F1,avg_best_F1s,std_best_F1s,'avg_best_F1',options);
        draw_result_v2(9,avg_best_Fp,avg_best_Fps,std_best_Fps,'avg_best_Fp',options);
end
end