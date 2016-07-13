function yHat = forward(X)
	% Assumption: X already include bias
	global W1 W2
	m = size(X, 1);
	z2 = X * W1;
	a2 = [ones(m, 1) sigmoid(z2)];
	z3 = a2 * W2;
	yHat = sigmoid(z3);
end