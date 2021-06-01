function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


regularization_param = 0;
for i=1:m
    %cost
    h_theta_x_i = sigmoid(theta'*X(i,:)');
    J = J+(-y(i)*log(h_theta_x_i) - (1-y(i))*log(1-h_theta_x_i));

    %calculate Gradient
    %cal with j = 0. in matlab, indexing starts from 1
    j = 1;
    grad(j)  = grad(j)+ (h_theta_x_i - y(i))*X(i,j);
    %cal with j>0
    for j=2:n
        grad(j)  = grad(j)+ (h_theta_x_i - y(i))*X(i,j);
    end
end 
%regularization parameters
for j=2:n
    regularization_param = regularization_param + theta(j).^2;
end

regularization_param = regularization_param*lambda/(2*m);

%divide to m
J = J/m;
J = J+ regularization_param;
grad = grad/m;
%cal finally grad
for j=2:n
    grad(j) = grad(j)+ + lambda*theta(j)/m;
end


% =============================================================

end
