function [perf_nmf,t] = run2pknn_nmf_feature(V_final,train_annot,test_annot)


train = V_final(1:4500,:);
test = V_final(4501:4999,:);

disp('caculate the distance matrix...');
t1=clock;
temp = pdist2(train, test);

mn = min(min(temp));
mx = max(max(temp));
norm_distances = (temp -  mn) / (mx - mn);

K1=5;
annotLabels = 5;
perf_nmf = zeros(30,4);
disp('run the 2pknn...');
for w=1:30
    disp(num2str(w));
    perf_nmf(w,:) = twopassknn(norm_distances,train_annot,test_annot,K1,w,annotLabels);
end
t2=clock;
t =etime(t2,t1);

end