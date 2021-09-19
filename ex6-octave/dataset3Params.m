function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
values_list = [ 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
n = size(values_list,1);

min_prediction_error =99999;
C_min = 0;
sigma_min = 0;
count = 0;
for i=1:n
   for j = 1:n
       C = values_list(i);
       sigma = values_list(j);
       
       model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
       predictions = svmPredict(model, Xval);
       prediction_error = mean(double(predictions ~= yval));
       if (prediction_error < min_prediction_error)
            C_min = C;
            sigma_min = sigma;
            min_prediction_error = prediction_error;
            %fprintf('\nPredict error = %.4f\n',prediction_error);
       end
       
   end    
end
%fprintf('\nSigma min = %.4f, C_min = %.4f\n,min prediction_error = %.4f',sigma_min, C_min,min_prediction_error);
C = C_min;
sigma = sigma_min;
% =========================================================================

end
