function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

temp_y  = y;
all_theta = all_theta';

training_steps = 1;
for epoch = 1:training_steps

    for label = 1:num_labels
        %
        y = temp_y == label;
        %calculate gradient 
        options = optimset('GradObj', 'on', 'MaxIter', 50);
        %
        initial_theta = all_theta(:,label);
        %[J grad] = lrCostFunction(all_theta(:,label), X, y, lambda);
        %[theta] = fmincg (@(t)(lrCostFunction(t, X, (y == label), lambda)), initial_theta, options);
        [theta] = fmincg (@(t)(lrCostFunction(t, X, y, lambda)), initial_theta, options);
        all_theta(:,label) = theta;
        %all_theta(:,label) = all_theta(:,label)-grad;
    end
end

all_theta = all_theta';
%{
training_steps = 1;
alpha = 0.1;
for epoch = 1:training_steps

    for label = 1:num_labels
        %
        y = temp_y == label;
        %calculate gradient 
        %fprintf('label: %f ,y(12) = %f\n', label,y(12));
        [J grad] = lrCostFunction(all_theta(:,label), X, y, lambda);
        %all_theta(:,label) = all_theta(:,label)-grad;
        all_theta(:,label) = all_theta(:,label)-alpha*grad;
    end
end
all_theta = all_theta';
%}


% =========================================================================


end
