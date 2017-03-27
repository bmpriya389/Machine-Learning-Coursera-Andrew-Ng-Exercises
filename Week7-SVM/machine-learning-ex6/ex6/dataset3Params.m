function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.




Cs = [0.01 0.03 0.1 0.3 1 3 10 30];
Sig2 = [0.01 0.03 0.1 0.3 1 3 10 30];
for c = 1:length(Cs)
    for s = 1:length(Sig2)
        C = Cs(c);
        sigma = Sig2(s);
        model = svmTrain(X, y, C,  @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error_terms(c,s) = mean(double(predictions ~= yval));
    end
end
 
min_error = min(min(error_terms));
[i, j] = find(error_terms == min_error);
C = Cs(i);
sigma = Sig2(j);




% =========================================================================

end
