function [X, y] = load_test_data()

loadNum = 32;

load(strcat('test_batches\', num2str(1), '.mat'));
X = affNISTdata.image;
y = affNISTdata.label_int;
for i = 2:loadNum
    load(strcat('test_batches\', num2str(i), '.mat'));
    X = [X, affNISTdata.image];
    y = [y, affNISTdata.label_int];
end

X = double(X)/255.0;
y = double(y);
end