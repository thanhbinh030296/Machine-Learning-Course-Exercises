function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m n ] = size(X);
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
regularized_grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
regularized_J = 0;

%cal regularized logistic regression
%cal regularized grad
for j=2:n
   regularized_J = regularized_J + theta(j).^2 ;
   
   regularized_grad(j)= regularized_grad(j) + theta(j);
end
%
%
for i=1:m
    %cost
    Xtheta = sigmoid(X(i,:)*theta);
    J = J - y(i)*log(Xtheta) - (1-y(i))*log(1-Xtheta);
    %calculate Gradient
    for j=1:n
        grad(j)  = grad(j)+ (Xtheta - y(i))*X(i,j);
    end
end 

%divide to m
J = J/m;
grad = grad/m;
regularized_J = regularized_J*lambda/(2*m);
regularized_grad = regularized_grad.*lambda/m;

J = J + regularized_J;
for j=2:n
    grad(j) = grad(j) + theta(j).*(lambda/m);
end

% =============================================================

grad = grad(:);

end
