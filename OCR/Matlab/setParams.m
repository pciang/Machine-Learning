function setParams(params)
	global input_layer_size hidden_layer_size output_layer_size
	global W1 W2
	W1_end = (input_layer_size + 1) * hidden_layer_size;
	W1 = reshape(params(1:W1_end), input_layer_size + 1, hidden_layer_size);
	W2 = reshape(params(W1_end + 1:end), hidden_layer_size + 1, output_layer_size);
end