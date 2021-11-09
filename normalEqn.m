function [theta] = normalEqn(X_train, y_train)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X_train, 2), 1);

theta = inv(X_train' * X_train) * X_train' * y_train; 


end
