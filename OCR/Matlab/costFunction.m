function cost = costFunction(X, y)
	% Assumption: X already include bias
	global m Lambda W1 W2
	yHat = forward(X);
	cost = sum(sum(-y .* log(yHat) - (1 - y) .* log(1 - yHat))) / m + Lambda * (sum(sum(W1(2:end, :) .^ 2)) + sum(sum(W2(2:end, :) .^ 2))) / (2. * m);
end