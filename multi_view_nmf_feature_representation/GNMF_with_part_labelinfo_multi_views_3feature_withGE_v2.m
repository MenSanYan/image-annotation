%here we select three different feature, gist\color\sift
% V4 = double(vec_read('corel5k_train_DenseSiftV3H1.hvecs'));%3000
V5 = double(vec_read('corel5k_train_Gist.fvec'));%512
% V10 = double(vec_read('corel5k_train_Hsv.hvecs32'));%4096


% t4 = double(vec_read('corel5k_test_DenseSiftV3H1.hvecs'));%3000
t5 = double(vec_read('corel5k_test_Gist.fvec'));%512
% t10 = double(vec_read('corel5k_test_Hsv.hvecs32'));%4096


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
%     V4(j,:) = V4(j,:)/norm(V4(j,:),2);
    V5(j,:) = V5(j,:)/norm(V5(j,:),1);
%     V10(j,:) = V10(j,:)/norm(V10(j,:),2);
end
%test data
for j=1:numOftest
%     t4(j,:) = t4(j,:)/norm(t4(j,:),2);
    t5(j,:) = t5(j,:)/norm(t5(j,:),1);
%     t10(j,:) = t10(j,:)/norm(t10(j,:),2);
end
%  test = [test5 ];%test10 test11 test12 test13 test14 test15


% X{1,1} = [V4;t4]';
X{1,1} = [V5;t5]';
% X{3,1} = [V10;t10]';

clear V5 t5;

%%semantic constrait
%
%
disp('caculate the sematic matrix...');
C = pdist2(T',T','cosine');
C = 1-C;

W_1 = T*C*T';
clear C T;
Wr = W_1;
Wi = W_1;
% matlabpool 1;
% for tu=2:2
    ti=1;
    Wi(W_1<=ti-1) = 1;
    Wi(W_1>ti-1) = 0;
    clear W_1;
    tu =  1.5;
    Wr(Wr<=tu) = 0;
    % Wr(Wr>2) = 1;
    %
%     for ti = 1:1

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
    scale = 100;
    options.alpha = scale*1;
    options.beta = scale*10^-4;
    options.error = 10^-4;
    options.nRepeat = 1;
%     options.gamma = 1;

%     alphas = [500,1000,1500,2000,2500,3000,3500];
%     betas = [0.01,0.1,1,10,100];
    disp('run the GNMF and run 2pknn...');
%     for j=1:7
%         for s=1:5
        iter = 1;
        
%         options.alpha = alphas(j);
%         options.beta = betas(s);
        results = cell(iter,1);
        ts = zeros(iter,1);
        k = 300;
        U = [];
        V = [];
        for i=1:iter
            tic;
            [U_final, V_final, nIter_final, objhistory_final] = GNMF_multi_views(X, k, Wr, Wi, options, U, V);
            disp('train time');
            toc;
            
            tic;
            [perf_nmf,t] = run2pknn_nmf_feature(V_final,train_annot,test_annot);
            disp('annotation time');
            toc;
            
            results{i,1} = perf_nmf;
            ts(i,1) = t;
        end
        savePath = '/home/xin/wzq/multi_view_nmf_feature_representation/tutlSelect/';
        saveName = ['results_seperatelyNMF_',num2str(options.alpha),'_beta',num2str(options.beta),'_L2_2_d',num2str(k),'_tu_',num2str(tu),'ti_',num2str(ti)];

%         save([savePath,saveName],'results');
%         end
%     end
    disp('end.');
%     end
% end
% matlabpool close;