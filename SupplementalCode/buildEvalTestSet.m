clear
clc

labelFilePrefix = './label/';
trainFilePrefix = './train/';

evalLabelSet = zeros(5,480,480);
evalSet = zeros(5,480,480);
testLabelSet = zeros(5,480,480);
testSet = zeros(5,480,480);

crop_vec = [(512-480)/2 (512-480)/2 479 479];
eval_test_vec = 3:3:30;

for idx = 1:5
    evalIdx = eval_test_vec(((idx-1)*2)+1);
   
    evalLabelFilename = strcat(labelFilePrefix,num2str(evalIdx),'.tif'); 
    evalFilename = strcat(trainFilePrefix,num2str(evalIdx),'.tif');
   
    evalLabelImg = imread(evalLabelFilename);
    evalImg = imread(evalFilename);  
    
    evalLabelSet(idx,:,:) = imcrop(evalLabelImg,crop_vec);
    evalSet(idx,:,:) = imcrop(evalImg,crop_vec);
   
    
    testIdx = eval_test_vec(idx*2);
   
    testLabelFilename = strcat(labelFilePrefix,num2str(testIdx),'.tif'); 
    testFilename = strcat(trainFilePrefix,num2str(testIdx),'.tif');
   
    testLabelImg = imread(testLabelFilename);
    testImg = imread(testFilename);  
    
    testLabelSet(idx,:,:) = imcrop(testLabelImg,crop_vec);
    testSet(idx,:,:) = imcrop(testImg,crop_vec);
end

evalLabelSet = evalLabelSet./255;
evalSet = evalSet./255;
testLabelSet = testLabelSet./255;
testSet = testSet./255;

save 'imgs_eval_test.mat' evalLabelSet evalSet testLabelSet testSet

for i = 1:size(evalSet,1)
    imagesc(reshape(evalSet(i,:,:),[480 480]))
    pause(0.5);
end

pause(1);

for i = 1:size(testSet,1)
    imagesc(reshape(testSet(i,:,:),[480 480]))
    pause(0.5);
end