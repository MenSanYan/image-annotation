train_annot = vec_read('corel5k_train_annot.hvecs');
test_annot = vec_read('corel5k_test_annot.hvecs');


V5 = double(vec_read('corel5k_train_Gist.fvec'));

 V10 = double(vec_read('corel5k_train_Hsv.hvecs32'));
 V11 = double(vec_read('corel5k_train_HsvV3H1.hvecs32'));
 V12 = double(vec_read('corel5k_train_Lab.hvecs32'));
 V13 = double(vec_read('corel5k_train_LabV3H1.hvecs32'));
 V14 = double(vec_read('corel5k_train_Rgb.hvecs32'));
 V15 = double(vec_read('corel5k_train_RgbV3H1.hvecs32'));

 train = [V15 ];%V10 V11 V12 V13 V14 V15
 
 test5 = double(vec_read('corel5k_test_Gist.fvec'));

 test10 = double(vec_read('corel5k_test_Hsv.hvecs32'));
 test11 = double(vec_read('corel5k_test_HsvV3H1.hvecs32'));
 test12 = double(vec_read('corel5k_test_Lab.hvecs32'));
 test13 = double(vec_read('corel5k_test_LabV3H1.hvecs32'));
 test14 = double(vec_read('corel5k_test_Rgb.hvecs32'));
 test15 = double(vec_read('corel5k_test_RgbV3H1.hvecs32'));
 
 test = [test15];% test10 test11 test12 test13 test14 test15

tic;
temp = pdist2(train, test);

mn = min(min(temp));
mx = max(max(temp));
norm_distances = (temp -  mn) / (mx - mn);

K1=5;
annotLabels = 5;
perf = zeros(30,4);
for w=1:15
    perf(w,:) = twopassknn(norm_distances,train_annot,test_annot,K1,w,annotLabels);
end
toc;