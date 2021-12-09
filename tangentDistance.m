% Get data ready

load('usps_resampled.mat');

testingLabels = [];
for i = 1:4649
    val = find(test_labels(:,i)==1);
    testingLabels(end+1) = val-1; %digits 0-9, not 1-10
end

trainingLabels = [];
for i = 1:4649
    val = find(train_labels(:,i)==1);
    trainingLabels(end+1) = val-1;
end

train_labels = testingLabels;
test_labels = trainingLabels;

train_data = train_patterns(:,1:270);
test_data = test_patterns(:,1:270);

len_train = length(train_data);
len_test = length(test_data);

% Calculate result of transformation functions

p_x_train = [];
p_y_train = [];
for i = 1:len_train
  image = reshape(train_data(:,i),[16,16])';
  [Gx,Gy] = imgradientxy(image);
  Gx_flat = reshape(Gx,[1,256]);
  Gy_flat = reshape(Gy,[1,256]);
  p_x_train = [p_x_train; Gx_flat];
  p_y_train = [p_y_train; Gy_flat];
end

p_x_test = [];
p_y_test = [];
for i = 1:len_test
  image = reshape(test_data(:,i),[16,16])';
  [Gx,Gy] = imgradientxy(image);
  Gx_flat = reshape(Gx,[1,256]);
  Gy_flat = reshape(Gy,[1,256]);
  p_x_test = [p_x_test; Gx_flat];
  p_y_test = [p_y_test; Gy_flat];
end

% Run TD algorithm

img1vec = p_x_train(1,:);
img1mat = reshape(img1vec,[16,16]);

A = [-p_x_train(1,:); p_x_test(1,:)]';
b = train_data(:,1) - test_data(:,1);
[x,flag,relres] = lsqr(A, b);


