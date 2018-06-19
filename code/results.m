clear all;
name = {'Cr','In','Pa','PS','RS','Sc'};
categories = {'Cr','In','Pa','PS','RS','Sc'};

%% Dairy setting
resultDir='~/NEU/results'
diaryFile='~/NEU/results/diary_NEU_CNN_conerned.txt'
diary(diaryFile) ; 
diary on ;

%% Load learnt CNN Model
net = load('~/NEU/code/imagenet-vgg-verydeep-19.mat') ;

%% Load the dataset
rootFolder = '~/NEU/database';
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

%% Initiate the matrices to store the results
c_mat_sum = zeros(length(categories),length(categories)); % confuse matrix
accuracy = zeros(10,1);
vali_matrix = cell(10,1);

%% Start
for h = 1:10
    % for official test 150 for testing 150 for training
    % [trainingSet, validationSet] = splitEachLabel(imds, 0.5, 'randomize');
    % for test code: 15 for testing 15 for training
    [imds_i, ~] = splitEachLabel(imds, 0.1, 'randomize'); % 300 in each categories, and 15 for train and 15 for test
    [trainingSet, validationSet] = splitEachLabel(imds_i, 0.5, 'randomize');
    trainingLabels = trainingSet.Labels;
    validationLabels = validationSet.Labels;
    
    %% Fearture Extraction on Test Set
    trainingFeatures = featureExtraction(dataset);
    
    %% SVM Classifier
    classifier = svmClassifier(trainingFeatures, TrainingLabels);
    
    %% Test on the validation Set
    validationFeatures = featureExtraction(dataset);
    predictedLabels = predict(classifier, validationSetFeatures);
    
    %% Check the accuracy
    labels = categorical(categories);
    num_class = length(labels); % number of different classes
    acc = zeros(nclass, 1); % accuracy for each class
    c_mat = zeros(num_class,num_class);
    vali_matrix{h,1} = cell(num_class,num_class);
    
    %% Calculate the accuracy of each class
    for i = 1 : num_class
        l = labels(i);
        idx = find(validationLabels == l); % for class labled l find its index
        curr_pred_label = predictedLabels(idx); % predicted lables
        curr_gnd_label = validationLabels(idx); % real label
        vali_i = vali_img(idx);
        acc(i) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
        for j = 1:num_class
            c_mat(i, j) = length(find(curr_pred_label == labels(j)))/length(idx);
            vali_matrix{h,1}{i,j} = vali_i(find(curr_pred_label == labels(j)));
        end
    end
    
    %% Mean of Testing Accuracy;
    accuracy(h,1) =  mean(acc); 
    fprintf('SVM: ')
    c_mat_sum += c_mat;
    fprintf('Classification accuracy of Linear SVM %d: %f\n', h, accuracy(h));
end

fprintf('Classification accuracy of Linear SVM %d: %f\n', mean(accuracy));

%% Plot the Confuction Matrix
c_mat_sum = c_mat_sum / 10; % Calculate the mean for 10 random test
figure
plotConfMat(c_mat_sum,categories);
save(fullfile(resultDir, sprintf('result-GLAC-NEU_CNN_conerned.mat')), ...
    'c_mat_sum','reflect','num_bin','ztol','def_rvecs','scale_of_rvecs','norm_flag','Diri_flag','feature','accuracy1');
print('-djpeg', fullfile(resultDir, sprintf('result-confusion-CNN-NEU.jpg')));
diary off;