function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X = [ones(m, 1) X];
%set K
K = 10;
%
A1 = X;

Z2 = A1*Theta1';
A2 = sigmoid(Z2);

%A2 = X_*Theta1_';
%A2 = [ones(m, 1) A2];
A2 = [ones(m, 1) A2];

Z3 = A2*Theta2';
A3 = sigmoid(Z3);
%convert y to Y multiple class
Y = zeros(m,K);

for i = 1:m
    Y(i,y(i)) = 1;
end
%
for i= 1:m
   for k = 1:K
      J = J + (-Y(i,k))*log(A3(i,k))-(1-Y(i,k))*log(1-(A3(i,k)));
   end
end
J = J/m;

%regularized cost function
% w_Theta1: Width of Theta1
% h_Theta1: Height of Theta1
% w_Theta2 and h_Theta2 are the same pattern
h_Theta1 = size(Theta1,1); w_Theta1 = size(Theta1,2);
h_Theta2 = size(Theta2,1); w_Theta2 = size(Theta2,2);

regularized_J = 0;
%add regularized of Theta1
for j = 1:h_Theta1
    for k = 2:w_Theta1
        regularized_J = regularized_J + Theta1(j,k).^2;
    end
end
%add regularized of Theta2
for j = 1:h_Theta2
    for k = 2:w_Theta2
        regularized_J = regularized_J + Theta2(j,k).^2;
    end
end
regularized_J = lambda*regularized_J/(2*m);

J = J+regularized_J;
%========================================================================================

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%error terms (delta)
error_terms3 =zeros(K,m);
for t = 1:m
    for k = 1:K
    %2. calculate in Layer 3
        error_terms3(k,t) = A3(k)-Y(t,k);
    end
end

%3. calculate for the hidden layer l=2
fprintf('\nsize of error_terms3: %d %d\n',size(error_terms3));


error_terms2 = Theta2'*error_terms3*sigmoidGradient([ones(m, 1) Z2]);
%add theta(1,0) = 1 to theta1


%
fprintf('\nsize of error_terms2: ',size(error_terms2),'\n');

fprintf('\nsize of A2: ',size(A2),'\n');

%4. calcuate delta
%delta2 = error_terms3*A3';
%delta2

%initial delta
delta2 = zeros(size(Theta2));
delta1 = zeros(size(Theta1));

delta2 =delta2+ error_terms3*A2;
fprintf('\nsize of delta2: ',size(delta2),'\n');

%
fprintf('\nsize of A1: ',size(A1),'\n');


%delta2 = delta2(2:end);
fprintf('\nSize of delta2 after 2:end: ',size(delta2));
size(delta2)
%5. Obtain the unregularized gradient

D2 = delta2./m;
%D1 = delta1./m;


%===========================================================================
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
