 disp('loading data');
 V5 = double(vec_read('corel5k_train_Gist.fvec'));

 V10 = double(vec_read('corel5k_train_Hsv.hvecs32'));
 V11 = double(vec_read('corel5k_train_HsvV3H1.hvecs32'));
 V12 = double(vec_read('corel5k_train_Lab.hvecs32'));
 V13 = double(vec_read('corel5k_train_LabV3H1.hvecs32'));
 V14 = double(vec_read('corel5k_train_Rgb.hvecs32'));
 V15 = double(vec_read('corel5k_train_RgbV3H1.hvecs32'));

 train = [V5 ];%V10 V11 V12 V13 V14 V15
 
 test5 = double(vec_read('corel5k_test_Gist.fvec'));

 test10 = double(vec_read('corel5k_test_Hsv.hvecs32'));
 test11 = double(vec_read('corel5k_test_HsvV3H1.hvecs32'));
 test12 = double(vec_read('corel5k_test_Lab.hvecs32'));
 test13 = double(vec_read('corel5k_test_LabV3H1.hvecs32'));
 test14 = double(vec_read('corel5k_test_Rgb.hvecs32'));
 test15 = double(vec_read('corel5k_test_RgbV3H1.hvecs32'));
 
 test = [test5 ];%test10 test11 test12 test13 test14 test15



% train = double(vec_read('corel5k_train_Gist.fvec'));
% test = double(vec_read('corel5k_test_Gist.fvec'));
train_annot = double(vec_read('corel5k_train_annot.hvecs'));
test_annot = double(vec_read('corel5k_test_annot.hvecs'));

X = train;
T = train_annot;
numOtrain = size(train,1);
numOtest = size(test,1);
%%semantic constrait
disp('caculate the semantic matrix....');
%
C = pdist2(T',T','cosine');
C = 1-C;

W_1 = T*C*T';
avg_W = sum(sum(W_1))/(numOtrain*numOtrain);
W_1 = exp(-W_1/(2*avg_W*avg_W));
W = W_1;
% W(W_1<=0.99) = 0;
% W(W_1>0.99) = 1;

[vals,ids] = sort(W,2);
knn = 10;
for i=1:numOtrain
   W(i,ids(i,knn:numOtrain)) = 0;
end


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

disp('dictionary learning for basis U_final from the train....');
k = 200;

%%dictionary learning for basis U_final from the train 
options.alpha = 0;


U = [];
V = [];

[U_train, V_train, nIter_train, objhistory_train] = GNMF(train', k, W, options, U, V);

%%encode the test feature based on the dictionary learned above.
disp('encode the test feature based on the dictionary learned above.'); 
options.alpha = 0;
[U_final, V_test, nIter_final, objhistory_final] = GNMF_V(test', k, W, options, U_train, V);

V_final = [V_train;V_test];
run2pknn_nmf_feature