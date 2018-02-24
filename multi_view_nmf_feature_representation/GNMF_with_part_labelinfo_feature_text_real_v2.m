%  disp('loading data...');
%  V5 = double(vec_read('corel5k_train_Gist.fvec'));
%  
%  test5 = double(vec_read('corel5k_test_Gist.fvec'));
% 
% train_annot = double(vec_read('corel5k_train_annot.hvecs'));
% test_annot = double(vec_read('corel5k_test_annot.hvecs'));
% 
% % X = cell(2,1);
% 
% 
% X{1,1} = [V5]';
% X{2,1} = [train_annot]';
%load the features that have get from the getAnnotationImage.m
disp('loading data...');

basePath = '/home/xin/wzq/corel5k/';
train_feature_files = dir(fullfile(basePath,'corel5k_train_*'));
test_feature_files = dir(fullfile(basePath,'corel5k_test_*'));
numOfviews = size(test_feature_files,1);
X = cell(1,1);

train_annot = double(vec_read('corel5k_train_annot.hvecs'));
test_annot = double(vec_read('corel5k_test_annot.hvecs'));

T = train_annot;
numOftrain = size(train_annot,1);
numOftest = size(test_annot,1);

% train = [];
% test = [];
% for i=1:5
%     %train data
%     train_tmp = double(vec_read(train_feature_files(i).name));
%     for j=1:numOftrain
%         train_tmp(j,:) = train_tmp(j,:)/norm(train_tmp(j,:),2);
%     end
%     train = [train,train_tmp];
%     %test data
%     test_tmp = double(vec_read(test_feature_files(i).name));
%     for j=1:numOftest
%         test_tmp(j,:) = test_tmp(j,:)/norm(test_tmp(j,:),2);
%     end
%     test = [test,test_tmp];
% end
    
%here we select three different feature, gist\color\sift
V4 = double(vec_read('corel5k_train_DenseSiftV3H1.hvecs'));%3000
V5 = double(vec_read('corel5k_train_Gist.fvec'));%512
V10 = double(vec_read('corel5k_train_Hsv.hvecs32'));%4096


t4 = double(vec_read('corel5k_test_DenseSiftV3H1.hvecs'));%3000
t5 = double(vec_read('corel5k_test_Gist.fvec'));%512
t10 = double(vec_read('corel5k_test_Hsv.hvecs32'));%4096

%train data
for j=1:numOftrain
    V4(j,:) = V4(j,:)/norm(V4(j,:),2);
%     V5(j,:) = V5(j,:)/norm(V5(j,:),2);
    V10(j,:) = V10(j,:)/norm(V10(j,:),2);
end
%test data
for j=1:numOftest
    t4(j,:) = t4(j,:)/norm(t4(j,:),2);
%     t5(j,:) = t5(j,:)/norm(t5(j,:),2);
    t10(j,:) = t10(j,:)/norm(t10(j,:),2);
end


train = [V4,V5,V10];
test = [t4,t5,t10];

X{1,1} = [train;test]';
% X{2,1} = train_annot';

% n = size(X,1);
% %%semantic constrait
% %
% %
% disp('caculate the sematic matrix...');
% C = pdist2(T',T','cosine');
% C = 1-C;
% 
% W_1 = T*C*T';
% avg_W = sum(sum(W_1))/(numOftrain*numOftrain);
% W_1 = exp(-W_1/(2*avg_W*avg_W));
% Wr = W_1;
% Wi = W_1;
% % W(W_1<=0.99) = 0;
% % W(W_1>0.99) = 1;
% 
% [vals,ids] = sort(Wr,2);
% knn = 10;
% for i=1:numOftrain
%    Wr(i,ids(i,knn:numOftrain)) = 0;
% end



%%feature manifold constrait
%
% 
% 
% W = pdist2(X,X);
% avg_W = sum(sum(W))/(n*n);
% 
% W = exp(-W/(2*avg_W*avg_W));
% % [vals,ids] = sort(W);
% 
% 
% W(W<=0.7) = 0;
% W(W>0.7) = 1;

% knn = 5;
% for i=1:n
%    W(i,ids(i,1:knn)) = 0;
% end
options.alpha = 0;
options.beta = 0;
% options.coefficient = [1];
options.error = 1e-5;
options.nRepeat = 5;

iter = 5;
for k=100:100:800
disp('run the GNMF and run 2pknn...');
results = cell(iter,1);
ts = zeros(iter,1);

U = [];
V = [];
Wr = [];
Wi = [];
% k = 500;
for i=1:iter
    [U_final, V_final, nIter_final, objhistory_final] = GNMF_multi_views(X, k, Wr, Wi, options, U, V);
%     [U_final_test, V_final_test, nIter_final_test, objhistory_final_test] = GNMF_V(test', k, Wr, options, U_final{1}, V);
%     tt = [V_final;V_final_test];
    tt = V_final;
    [perf_nmf,t] = run2pknn_nmf_feature(tt,train_annot,test_annot);
%     [perf_nmf,t] = run2pknn_nmf_feature_my(tt,train_annot,test_annot);
    results{i,1} = perf_nmf;
    ts(i,1) = t;
end

savePath = '/home/xin/wzq/multi_view_nmf_feature_representation/results_comp/';
saveName = ['results_concatenteNMF_alpha',num2str(options.alpha),'_beta',num2str(options.beta),'_L2_2_d',num2str(k)];

save([savePath,saveName],'results');
end
disp('end.');