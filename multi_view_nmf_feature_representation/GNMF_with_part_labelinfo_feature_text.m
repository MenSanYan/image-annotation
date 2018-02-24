 disp('loading data...');
 V5 = double(vec_read('corel5k_train_Gist.fvec'));
 
 test5 = double(vec_read('corel5k_test_Gist.fvec'));

train_annot = double(vec_read('corel5k_train_annot.hvecs'));
test_annot = double(vec_read('corel5k_test_annot.hvecs'));

X = cell(2,1);


X{1,1} = [V5;test5]';
X{2,1} = [train_annot;test_annot]';



T = train_annot;
n = size(X,1);
numOftrain = size(V5,1);
numOftest = size(test5,1);
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

disp('run the GNMF and run 2pknn...');
iter = 3;
results = cell(iter,1);
ts = zeros(iter,1);

U = [];
V = [];
for i=1:iter
    k = 300;
    [U_final, V_final, nIter_final, objhistory_final] = GNMF_multi_views(X, k, Wr, Wi, options, U, V);
    [perf_nmf,t] = run2pknn_nmf_feature(V_final,train_annot,test_annot);
    results{i,1} = perf_nmf;
    ts(i,1) = t;
end

disp('end.');