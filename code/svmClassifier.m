function classifier = svmClassifier(features, labels)
  % features are the results from CNN
  % labels are the ground truth of the data
   
  % fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
  tmp = templateSVM('Standardize',1,'KernelFunction','linear'); % polynomial, linear svm
  classifier = fitcecoc(features, labels, 'Learners', tmp, 'Coding', 'onevsall');
endfunction
