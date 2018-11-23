function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = -((sum(((y .* log(sigmoid(X*theta))) - ((y-1) .* log(1 - sigmoid(X*theta))))))/m) ...
    + sum(theta(2:end).^2)*lambda/(2*m);

%grad(1) = sum(((sigmoid(X(:,1)*theta(1)))-y(:,1)).*X(:,1)) / m;
%grad(2:end) = sum(((sigmoid(X(:,2:end)*theta(2:end)))-y(:,2:end)).*X(:,2:end)) / m + lambda * theta(2:end) / m;

grad = sum(((sigmoid(X*theta))-y).*X) / m;
grad(2:end) = grad(2:end) + (lambda * theta(2:end))' / m;


% =============================================================

end
