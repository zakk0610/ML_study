function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% train data is row base 
% theta is row base
h = X * theta;
% set bias's theta as 0
theta_0 = theta;
theta_0(1)=0;

J = 1/(2*m) * sum(((h-y).^2)(:)) + lambda/(2*m) * sum((theta_0.^2)(:));

% sum all training data's delta
% repmat mean extend someones's column 
% ex. somones = [7;6;5] repmat(someone, 1, 4)
% result is [7,7,7,7; 6,6,6,6; 5,5,5,5] 
grad = (1/m) * sum(repmat((h - y), 1, size(X,2)).*X)' + theta_0.*(lambda/m);

% =========================================================================

grad = grad(:);

end
