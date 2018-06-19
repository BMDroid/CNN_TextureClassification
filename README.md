# CNN_TextureClassification
Texture Classification using Convolutional Neural Networks.

1. The **pretrained CNN model** in this project is not in the repository.

You may not need to download it but according to [MathWorks](https://www.mathworks.com/help/nnet/ref/vgg19.html),  you can use it this way:

```matlab
net = vgg19;
```

2. I use **Linear SVM** in the project.

   ```matlab
   % fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
     tmp = templateSVM('Standardize',1,'KernelFunction','linear'); % polynomial, linear svm
     classifier = fitcecoc(features, labels, 'Learners', tmp, 'Coding', 'onevsall');
   ```

3. The result:

   <p align="center">  <img src="https://github.com/BMDroid/CNN_TextureClassification/blob/master/results/result-confusion-CNN-NEU.jpg" width="50%">
   </p>  

4. Conclusion

   CNN has great performance on texture classification.