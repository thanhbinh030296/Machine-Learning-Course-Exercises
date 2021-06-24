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
X_ = X;
X_ = [ones(m, 1) X_];

%set K
K = num_labels;
%
A1 = X_;

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
%y = Y;
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


%error terms 
%error_terms3 =zeros(K,1);
error_terms3 =zeros(K,1);
%delta1 = zeros(hidden_layer_size, input_layer_size);
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
%
error_terms2 = zeros(hidden_layer_size,1);
%

for t = 1:m
    A_1 = X(t,:);
    A_2 = sigmoid([1 A_1]*Theta1');
    A_3 = sigmoid([1 A_2]*Theta2');
    for k = 1:K
    %2. calculate in Layer 3
        error_terms3(k) = A_3(k)-Y(t,k);
    end
    %3. calcute for the hidden layer l = 2
    %fprintf('\nsize of error_terms 3: %d %d',size(error_terms3));
    
    error_terms2 = Theta2'*error_terms3 .*sigmoidGradient([1 [1 A_1]*Theta1'])';
    
    %fprintf('\nsize of A2 : %d %d\n', size(sigmoidGradient([1 [1 A_1]*Theta1'])'));
    %remove error_term 2 at index 0
    error_terms2 = error_terms2(2:end);
    %
    %fprintf('\nsize of error_terms2 %d %d\n',size(error_terms2));
    %fprintf('\nsize of A_1 %d %d\n',size(A_1));
    %
    delta2 = delta2 + error_terms3*[1 A_2];
    %fprintf('\nsize of delta 2 : %d %d\n',size(delta2));
    %fprintf('\nsize of delta1 : %d %d \n',size(delta1));
    delta1 = delta1 + error_terms2*[1 A_1];  
end




%===========================================================================
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%unregularized gradient 
D1 = delta1./m;
D2 = delta2./m;

%reguliarized gradient
for i = 1:h_Theta1
    for j = 2:w_Theta1
        D1(i,j) = D1(i,j) + lambda*Theta1(i,j)/m;
    end
end
%
for i = 1:h_Theta2
    for j = 2:w_Theta2
        D2(i,j) = D2(i,j) + lambda*Theta2(i,j)/m;
    end
end

Theta1_grad =D1;
Theta2_grad =D2;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
