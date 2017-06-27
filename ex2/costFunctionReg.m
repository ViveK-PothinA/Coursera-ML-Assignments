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

htheta = sigmoid(X*theta);
diff1 = (-y).*log(htheta);
diff2 = (1-y).*log(1-htheta);
differ = diff1 - diff2;

theta2 = theta.^2
theta2(1,1) = 0; %Should not consider theta(1)

summ = (lambda/(2*m))*sum(theta2,1)

J = (1/m)*(sum(differ)) + summ
theta(1,1) = 0;
grad = (1/m).*((X')*(htheta - y)) + (lambda/m)*theta




% =============================================================

end
