clear
clc

labelFilePrefix = './label/';
trainFilePrefix = './train/';

labelSet = [];
trainSet = [];

for imgIdx = 1:30
   labelFilename = strcat(labelFilePrefix,num2str(imgIdx),'.tif'); 
   trainFilename = strcat(trainFilePrefix,num2str(imgIdx),'.tif');
   [labelSetNew,trainSetNew] = generateTrainingSet(labelFilename,trainFilename);
   labelSet = cat(1,labelSet,labelSetNew);
   trainSet = cat(1,trainSet,trainSetNew);
end

labelSet = labelSet./255;
trainSet = trainSet./255;

save 'imgs_mask_train_large.mat' labelSet
save 'imgs_train_large.mat' trainSet