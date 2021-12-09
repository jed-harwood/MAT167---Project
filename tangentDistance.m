% Get data ready

load('usps_resampled.mat');

testingLabels = [];
for i = 1:4649
    val = find(test_labels(:,i)==1);
    testingLabels(end+1) = val-1;
end

trainingLabels = [];
for i = 1:4649
    val = find(train_labels(:,i)==1);
    trainingLabels(end+1) = val-1;
end

train_labels = testingLabels;
test_labels = trainingLabels;

train_data = train_patterns(:,1:500);
test_data = test_patterns(:,1:500);

len_train = length(train_data);
len_test = length(test_data);

% Calculate result of transformation functions
tic
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

predictions = [];
for t = 1:len_test

    residuals = [];
    for r = 1:len_train
        A = [-p_x_train(r,:); p_x_test(t,:)]';
        b = train_data(:,r) - test_data(:,t);
        [x,flag,relres] = lsqr(A, b);
  
        res = (norm(b - A*x))^2;
        residuals = [residuals res];
    end
    
    [min_resid,index] = min(residuals);
    disp(min_resid)
    predictions = [predictions train_labels(index)];
end
toc

% Calculate accuracy
correct = 0;
for i = 1:len_test
    pred = predictions(i);
    true = test_labels(i);
    if pred == true
        correct = correct + 1;
    end
end
percentage = correct/len_test;