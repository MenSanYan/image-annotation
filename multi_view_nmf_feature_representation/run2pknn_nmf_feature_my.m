function [perf_nmf,t] = run2pknn_nmf_feature_my(V_final,train_annot,test_annot)


train = V_final(1:4500,:);
test = V_final(4501:4999,:);

disp('caculate the distance matrix...');
t1=clock;
% temp = pdist2(test,train);
% 
% mn = min(min(temp));
% mx = max(max(temp));
% norm_distances = (temp -  mn) / (mx - mn);

W = pdist2(test,train,'euclidean');
% W = exp(-W);
for i=1:499 
    W(i,:) = W(i,:)/norm(W(i,:),2);
end
norm_distances = W;

% K=10;
% annotLabels = 5;
% disp('find the K nearest neighbor and asign label...');
% [~,ids] =sort(norm_distances,2);
% label_predict = zeros(499,260);
% for i=1:499
%     labels = sum(train_annot(ids(i,1:K),:));
%     [~,lids] = sort(labels,2,'descend');
%     label_predict(i,lids(1:annotLabels)) = 1;
% end
% perf_nmf = performance_com(test_annot,label_predict);


K=5;
annotLabels = 4;
disp('find the K nearest neighbor and asign label...');
label_predict = zeros(499,260);
[~,ids] = sort(norm_distances,2);%,'descend'
for i=1:499
    labels_weighted = sum(train_annot(ids(i,1:K),:));%norm_distances(i,ids(i,1:K))*
    [~,lids] = sort(labels_weighted,'descend');
    label_predict(i,lids(1:annotLabels)) = 1;
end
perf_nmf = performance_com(test_annot,label_predict);


t2=clock;
t =etime(t2,t1);

end