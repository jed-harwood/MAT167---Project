
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

% Process data
% do smoothing here
% tuples = [];
% for i = 1:16
%     for j = 1:16
%         tuples = [tuples; i j];
%     end
% end
% 
% tuples_x = [];
% tuples_y = [];
% for l = 256
%     tuples_x = [tuples_x; tuples(l,1)];
%     tuples_y = [tuples_x; tuples(l,2)];
% end
% 
% for t = 1:len_train
%     image = reshape(train_data(:,t),[16,16])';
%     S = [];
%     for tup = 1:size(tuples,1)
%         i = tuples(i,1);
%         j = tuples(i,2);
%         S_gaussian = image(i,j)* exp(1).^(-((tuples_x-i).^2+(tuples_y-j).^2)/(2*0.9^2));
%         S(t) = sum(S_gaussian);
%     end
% end
% 
% for i = 1:len_test
% 
% end

% Get values x, y partial deriv components

p_x_train = [];
p_y_train = [];
for i = 1:len_train
  image = reshape(train_data(:,i),[16,16])';
  G = gradient(image);
  p_x_train = [p_x_train; G(1,:)];
  p_y_train = [p_y_train; G(1,:)];
end

p_x_test = [];
p_y_test = [];
for i = 1:len_test
  image = reshape(train_data(:,i),[16,16])';
  G = gradient(image);
  p_x_test = [p_x_test; G(1,:)];
  p_y_test = [p_y_test; G(2,:)];
end

% Run algorithm

predictions = [];
for p = 1:len_test

    A = [];
    b = [];
    for i = 1:len_train
        A_val = horzcat(-p_x_train(i,:), p_x_test(i,:));
        b_val = train_data(:,i) - train_data(:,p);
        A = [A; A_val];
        b = [b; b_val];
    end

    residuals = [];
    xs = [];
    for i = 1:len_train
        A_row = A(i,:);
        b_row = b(i,:);
        [x,flag,relres] = lsqr(A_row, b_row);
        residuals(i) = abs(A_row * x - b_row);
    end

    [min_resid,index] = min(residuals);
    predictions(i) = train_labels(index);
end

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

