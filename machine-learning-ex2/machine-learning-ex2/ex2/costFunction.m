function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
[m n ] = size(X);
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for i=1:m
    %cost
    h_theta_x_i = sigmoid(theta'*X(i,:)');
    J = J - y(i)*log(h_theta_x_i) - (1-y(i))*log(1-h_theta_x_i);
    %calculate Gradient
    for j=1:n
        grad(j)  = grad(j)+ (h_theta_x_i - y(i))*X(i,j);
    end
end 
%divide to m
J = J/m;
grad = grad/m;

% =============================================================

end
