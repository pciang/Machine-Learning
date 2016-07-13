function result = sigmoid(z)
	result = 1. ./ (1. + exp(-z));
end