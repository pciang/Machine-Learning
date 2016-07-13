clear; close all; clc

f_in = fopen('data.txt', 'r');

digit = 0;

global X y
X = 1:400;
y = 1:10;

global m
m = 0;

while ~feof(f_in)
	line = fgetl(f_in);
	if numel(line) == 1
		digit = str2num(line);
		continue
	end
	
	m = m + 1;
	temp = strsplit(line, ',');
	X = [X; str2double(temp)];
	y = [y; 0:9 == digit];
end

X = [ones(m, 1) X(2:end, :)];
y = y(2:end, :);

global input_layer_size hidden_layer_size output_layer_size

% does not include bias
input_layer_size = 400;
hidden_layer_size = 25;
output_layer_size = 10;

global W1 W2 Lambda
W1 = rand(input_layer_size + 1, hidden_layer_size) * 2. - 1.;
W2 = rand(hidden_layer_size + 1, output_layer_size) * 2. - 1.;
Lambda = 0.01;

options = optimset('MaxIter', 100);
[params, cost] = fmincg(@costFunctionWrapper, flattenParams(), options);

setParams(params)

for digit = 0:9
	fprintf('\nDigit-%d:\n', digit)
	for tc = 0:100
		filename = sprintf('../data/test%d_%d.png', digit, tc);
		try
			[A, map] = imread(filename);
			fprintf('Reading file "%s"\n', filename)
			dataX = 1. - im2double(A(:, :, 1));
			dataX = [1 dataX(:)'];
			result = forward(dataX);
			[maxval, maxind] = max(result);
			fprintf('Predicted: %d\n', maxind - 1)
		catch ME
			% disp(ME)
			break
		end
	end
end