function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

n = size(X,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%fprintf('\nm = %d N = %d',m,n);
% calculate J
for i = 1:m
   h_theta_xi = theta'*X(i,:)';
   J = J+ (h_theta_xi - y(i)).^2;
end

regularized_J = 0;

if lambda >0
    for j =2:n
        regularized_J = regularized_J + theta(j)^2;
    end
end
regularized_J = regularized_J*lambda;

J = J+ regularized_J;

J = J./(2*m);


%==========================================================================
% calculate Gradient

for j=1:n
    for i = 1:m
        h_theta_xi = theta'*X(i,:)';
        grad(j) = grad(j) + (h_theta_xi - y(i))*X(i,j);
    end
end    
%addition regularization parameters

for j=2:n
   grad(j) = grad(j)+ lambda*theta(j);
end

grad = grad.*(1/m);
% =========================================================================

grad = grad(:);

end
