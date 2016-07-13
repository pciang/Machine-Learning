function numgrad = checkGradient(X, y)
	params_init = flattenParams();
	numgrad = zeros(size(params_init));
	perturb = zeros(size(params_init));
	e = 1e-4;
	
	for i = 1:numel(params_init)
		perturb(i) = e;
		setParams(params_init + perturb);
		loss1 = costFunction(X, y);
		
		setParams(params_init - perturb);
		loss2 = costFunction(X, y);
		
		numgrad(i) = (loss1 - loss2) / (2. * e);
		perturb(i) = 0;
	end
	
	setParams(params_init);
end