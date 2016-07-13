function result = sigmoidPrime(z)
	result = sigmoid(z) .* (1 - sigmoid(z));
end