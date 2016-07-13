function [grad1, grad2] = backprop(X, y)
	% Assumption: X already include bias
	
	global W1 W2
	grad1 = zeros(size(W1));
	grad2 = zeros(size(W2));
	
	global m Lambda
	z2 = X * W1;
	a2_nobias = sigmoid(z2);
	a2 = [ones(m, 1) a2_nobias];
	z3 = a2 * W2;
	a3 = sigmoid(z3);
	
	delta3 = a3 - y;
	delta2 = delta3 * W2(2:end, :)' .* sigmoidPrime(z2);
	grad2 = (grad2 + a2' * delta3) ./ m;
	grad1 = (grad1 + X' * delta2) ./ m;
	grad1(2:end, :) = grad1(2:end, :) + Lambda * grad1(2:end, :) ./ m;
	grad2(2:end, :) = grad2(2:end, :) + Lambda * grad2(2:end, :) ./ m;
end