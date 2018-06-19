 function features = featureExtraction(dataset)
   % size of the dataset
   num = numel(dataset.Files);
   % features
   features = [];
   % empty matrix for storing all the features from CNN
   hog = [];
   % extraction the feature
   for i = 1:num
     img = readimage(dataset, i);
     img = repmat(img, 1, 1, 3); % replicate grayscale image to 3 dimension to fit CNN input 
     img = single(img_i);
     img = imresize(img, net.normalization.imageSize(1:2)); % resize the image
     hog = vl_simplenn(net, img); 
     features(i,:) = hog(42).x(:,:,1:end); % only extract the 42th layer results
   endfor
 endfunction