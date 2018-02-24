%load the features that have get from the getAnnotationImage.m
disp('loading data...');

basePath = '/home/xin/wzq/corel5k/';
train_feature_files = dir(fullfile(basePath,'corel5k_train_*'));
test_feature_files = dir(fullfile(basePath,'corel5k_test_*'));

train_annot = double(vec_read('corel5k_train_annot.hvecs'));
test_annot = double(vec_read('corel5k_test_annot.hvecs'));

T = train_annot;
numOftrain = size(train_annot,1);
numOftest = size(test_annot,1);

numOfviews = size(test_feature_files,1);
X = cell(9,1);
for i=1:9
    %train data
    fileName = train_feature_files(i).name;
    if ~isempty(strfind(fileName,'hvecs'))
        %X^2
        train_tmp = double(vec_read(fileName));
%         for j=1:numOftrain
%             train_tmp(j,:) = train_tmp(j,:)/norm(train_tmp(j,:),2);
%         end  
        %test data
        test_tmp = double(vec_read(test_feature_files(i).name));
%         for j=1:numOftest
%             test_tmp(j,:) = test_tmp(j,:)/norm(test_tmp(j,:),2);
%         end
        X{i,1} = [train_tmp;test_tmp]';
    elseif ~isempty(strfind(fileName,'hvecs32'))
        %L1
        train_tmp = double(vec_read(fileName));
%         for j=1:numOftrain
%             train_tmp(j,:) = train_tmp(j,:)/norm(train_tmp(j,:),2);
%         end  
        %test data
        test_tmp = double(vec_read(test_feature_files(i).name));
%         for j=1:numOftest
%             test_tmp(j,:) = test_tmp(j,:)/norm(test_tmp(j,:),2);
%         end
        X{i,1} = [train_tmp;test_tmp]';
    else
        %L2
        train_tmp = double(vec_read(fileName));
%         for j=1:numOftrain
%             train_tmp(j,:) = train_tmp(j,:)/norm(train_tmp(j,:),2);
%         end  
        %test data
        test_tmp = double(vec_read(test_feature_files(i).name));
%         for j=1:numOftest
%             test_tmp(j,:) = test_tmp(j,:)/norm(test_tmp(j,:),2);
%         end
        X{i,1} = [train_tmp;test_tmp]';
    end
end

%%semantic constrait
%
%
disp('caculate the sematic matrix...');
C = pdist2(T',T','cosine');
C = 1-C;

W_1 = T*C*T';
% avg_W = sum(sum(W_1))/(numOftrain*numOftrain);
% W_1 = exp(-W_1/(2*avg_W*avg_W));
Wr = W_1;
Wi = W_1;
Wr(Wr<=2) = 0;
% Wr(Wr>2) = 1;
% 
Wi(Wi==0) = 1;
Wi(Wi~=1) = 0;



% [vals,ids] = sort(Wr,2);
% knn = 200;
% for i=1:numOftrain
%    Wr(i,ids(i,knn:numOftrain)) = 0;
% end

% knn2 = 200;
% for i=1:numOftrain
%    Wi(i,ids(i,numOftrain-knn2+1:numOftrain)) = 1;
%    Wi(i,ids(i,1:numOftrain-knn2)) = 0;
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
options.nRepeat = 3;
options.error = 1e-4;

disp('run the GNMF and run 2pknn...');
iter = 3;
results = cell(iter,1);
ts = zeros(iter,1);

U = [];
V = [];
for i=1:iter
    k = 1000;
    [U_final, V_final, nIter_final, objhistory_final] = GNMF_multi_views(X, k, Wr, Wi, options, U, V);
    [perf_nmf,t] = run2pknn_nmf_feature(V_final,train_annot,test_annot);
    results{i,1} = perf_nmf;
    ts(i,1) = t;
end

disp('end.');