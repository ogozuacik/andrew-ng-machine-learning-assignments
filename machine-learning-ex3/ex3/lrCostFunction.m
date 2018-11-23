function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = -((sum(((y .* log(sigmoid(X*theta))) - ((y-1) .* log(1 - sigmoid(X*theta))))))/m) ...
    + sum(theta(2:end).^2)*lambda/(2*m);

grad = sum(((sigmoid(X*theta))-y).*X) / m;
grad(2:end) = grad(2:end) + (lambda * theta(2:end))' / m;

% =============================================================

grad = grad(:);

end
