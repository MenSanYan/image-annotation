

    
 V1 = double(vec_read('corel5k_test_DenseHue.hvecs'));%100
 V2 = double(vec_read('corel5k_test_DenseHueV3H1.hvecs'));%300
 V3 = double(vec_read('corel5k_test_DenseSift.hvecs'));%1000
 V4 = double(vec_read('corel5k_test_DenseSiftV3H1.hvecs'));%3000
 %row's sum = 1024
 
 V5 = double(vec_read('corel5k_train_Gist.fvec'));%512
 
 
 V6 = double(vec_read('corel5k_test_HarrisHue.hvecs'));%100
 V7 = double(vec_read('corel5k_test_HarrisHueV3H1.hvecs'));%300
 V8 = double(vec_read('corel5k_test_HarrisSift.hvecs'));%1000
 V9 = double(vec_read('corel5k_test_HarrisSiftV3H1.hvecs'));%3000
 
 
 %row's sum = yizhi exist zero row.
 
 V10 = double(vec_read('corel5k_train_Hsv.hvecs32'));%4096
 V11 = double(vec_read('corel5k_train_HsvV3H1.hvecs32'));%5184
 V12 = double(vec_read('corel5k_train_Lab.hvecs32'));%4096
 V13 = double(vec_read('corel5k_train_LabV3H1.hvecs32'));%5184
 V14 = double(vec_read('corel5k_train_Rgb.hvecs32'));%4096
 V15 = double(vec_read('corel5k_train_RgbV3H1.hvecs32'));%5184
 %row's sum =98304