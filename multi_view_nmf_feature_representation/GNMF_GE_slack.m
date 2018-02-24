%here we select three different feature, gist\color\sift
V4 = double(vec_read('corel5k_train_DenseSiftV3H1.hvecs'));%3000
V5 = double(vec_read('corel5k_train_Gist.fvec'));%512
V10 = double(vec_read('corel5k_train_Hsv.hvecs32'));%4096


t4 = double(vec_read('corel5k_test_DenseSiftV3H1.hvecs'));%3000
t5 = double(vec_read('corel5k_test_Gist.fvec'));%512
t10 = double(vec_read('corel5k_test_Hsv.hvecs32'));%4096


% train = double(vec_read('corel5k_train_Gist.fvec'));
% test = double(vec_read('corel5k_test_Gist.fvec'));
train_annot = double(vec_read('corel5k_train_annot.hvecs'));
test_annot = double(vec_read('corel5k_test_annot.hvecs'));

X = cell(3,1);
T = train_annot;
n = size(X,1);
numOftrain = size(train_annot,1);
numOftest = size(test_annot,1);



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
%  test = [test5 ];%test10 test11 test12 test13 test14 test15


X{1,1} = [V4;t4]';
X{2,1} = [V5;t5]';
X{3,1} = [V10;t10]';
%%semantic constrait
%
%
disp('caculate the sematic matrix...');
C = pdist2(T',T','cosine');
C = 1-C;

W_1 = T*C*T';
avg_W = sum(sum(W_1))/(numOftrain*numOftrain);
W_1 = exp(-W_1/(2*avg_W*avg_W));
Wr = W_1;
% W(W_1<=0.99) = 0;
% W(W_1>0.99) = 1;

[vals,ids] = sort(Wr,2);
knn = 10;
for i=1:numOftrain
   Wr(i,ids(i,knn:numOftrain)) = 0;
end

Wi = [];

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
options.gamma = 10;
options.nRepeat = 1;
options.error = 10^-4;
disp('run the GNMF and run 2pknn...');
rangeD = [0.01,100,200,1000];
for d=1:4
    
    options.delta = rangeD(d);

    iter = 1;
    results = cell(iter,1);
    ts = zeros(iter,1);

    U = [];
    V = [];
    for i=1:iter
        k = 300;
        [U_final, V_final, Vx_final, nIter_final, objhistory_final,coefficient] = GNMF_multi_views_slack_GE(X, k, Wr, Wi, options, U, V);
        [perf_nmf,t] = run2pknn_nmf_feature(Vx_final,train_annot,test_annot);
        results{i,1} = perf_nmf;
        ts(i,1) = t;
    end

    savePath = '/home/xin/wzq/multi_view_nmf_feature_representation/results_comp_slack/';
    saveName = ['results_slackNMFwithLearningCoefficient_alpha',num2str(options.alpha),'_beta',num2str(options.beta),'_delta',num2str(d),'_L2_2_d',num2str(k)];

    save([savePath,saveName],'results');
end
disp('end.');