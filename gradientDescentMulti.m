function [theta, J_history] = gradientDescentMulti(X_train, y_train, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
% Initialize some useful values
m = length(y_train); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    h = X_train * theta;
    theta = theta - ((alpha/m) * (X_train'* (h-y_train))); 
    
    
    
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X_train, y_train, theta);
end
end
