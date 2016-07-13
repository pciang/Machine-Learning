function [cost, grads] = costFunctionWrapper(params)
	global X y
	setParams(params)
	
	cost = costFunction(X, y);
	
	[g1, g2] = backprop(X, y);
	grads = [g1(:); g2(:)];
end