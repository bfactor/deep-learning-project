function [labelSet,trainSet] = generateTrainingSet(labelFilename,trainFilename)
    labelImg = imread(labelFilename);
    trainImg = imread(trainFilename);
    
    crop_vec = [(512-480)/2 (512-480)/2 479 479]; 
    labelSet = zeros(18,480,480);
    trainSet = zeros(18,480,480);
    
    
    %Crop original images
    labelSet(1,:,:) = imcrop(labelImg,crop_vec);
    trainSet(1,:,:) = imcrop(trainImg,crop_vec);
    
    % Generate 7 rotated images
    rot_vec = [-1 -90 -91 -180 -181 -270 -271];
    sidx = 2;
    for idx = 1:size(rot_vec,2)
       labelSet(sidx+idx-1,:,:) = ...
           imcrop(imrotate(labelImg,rot_vec(idx),'crop'),crop_vec);
       trainSet(sidx+idx-1,:,:) = ...
           imcrop(imrotate(trainImg,rot_vec(idx),'crop'),crop_vec); 
    end
    
    % Generate 2 mirrored images
    sidx = sidx + size(rot_vec,2);
    flip_vec = [1 2];
    for idx = 1:size(flip_vec,2)
       labelSet(sidx+idx-1,:,:) = ...
           imcrop(flip(labelImg,flip_vec(idx)),crop_vec);
       trainSet(sidx+idx-1,:,:) = ...
           imcrop(flip(trainImg,flip_vec(idx)),crop_vec); 
    end
    
    % Generate 8 translated images
    sidx = sidx + size(flip_vec,2);
    trans_vec = [16 -16 0 0 16 16 -16 -16; 0 0 16 -16 16 -16 16 -16];
    for idx = 1:size(trans_vec,2)
       labelSet(sidx+idx-1,:,:) = ...
           imcrop(imtranslate(labelImg,trans_vec(:,idx)),crop_vec);
       trainSet(sidx+idx-1,:,:) = ...
           imcrop(imtranslate(trainImg,trans_vec(:,idx)),crop_vec);     
    end
end