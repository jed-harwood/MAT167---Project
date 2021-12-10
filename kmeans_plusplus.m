load('usps_resampled.mat'); 

%16x16 images, 9298 images total 

%test_labels 10x4649 (1 if number, -1 else)
%test_patterns 256*4649

%train_labels 10x4649 (1 if number, -1 else)
%train_patterns 256*4649


%obtain all testing labels
testingLabels = [];
for i = 1:4649
    val = find(test_labels(:,i)==1);
    testingLabels(end+1) = val-1; %digits 0-9, not 1-10
end

%obtain all training labels
trainingLabels = [];
for i = 1:4649
    val = find(train_labels(:,i)==1);
    trainingLabels(end+1) = val-1;
end

%column indicies for each digit in train_labels
zeros_index = find(trainingLabels == 0);
ones_index = find(trainingLabels == 1);
twos_index = find(trainingLabels == 2);
threes_index = find(trainingLabels == 3);
fours_index = find(trainingLabels == 4);
fives_index = find(trainingLabels == 5);
sixs_index = find(trainingLabels == 6);
sevens_index = find(trainingLabels == 7);
eigths_index = find(trainingLabels == 8);
nines_index = find(trainingLabels == 9);

%columns for each digit
zero = train_patterns(:,zeros_index);
ones = train_patterns(:,ones_index);
twos = train_patterns(:,twos_index);
threes = train_patterns(:,threes_index);
fours = train_patterns(:,fours_index);
fives = train_patterns(:,fives_index);
sixs = train_patterns(:,sixs_index);
sevens = train_patterns(:,sevens_index);
eigths = train_patterns(:,eigths_index);
nines = train_patterns(:,nines_index);

%mean for each digit 
means = zeros(256,10);
means(:,1) = mean(zero,2);
means(:,2) = mean(ones,2);
means(:,3) = mean(twos,2);
means(:,4) = mean(threes,2);
means(:,5) = mean(fours,2);
means(:,6) = mean(fives,2);
means(:,7) = mean(sixs,2);
means(:,8) = mean(sevens,2);
means(:,9) = mean(eigths,2);
means(:,10) = mean(nines,2);

%-----------------------Where K-means begins-----------------------------

%idx = for each data point, the cluster number its in (4649x1)
% class_kmeans(i)  = digit classification for the cluster 

%set seed for initial cluster random locations
accuracies_plusplus = nan(1,7);
runtime_plusplus = nan(1,7);
for s = 1:7
    
    rng(s);
    tic %run time
    k = 10;
    [idx,C] = kmeans(transpose(test_patterns),k); 
    % C final centroid locations 10x256
    
    % Assigning clusters to digit
    C_256x10 = C'; %reshape
    class_kmeans = NaN(10,1); % store classification should be 10x1
    for i=1:10
        J = Inf; % initialize large distance between centroids
        label = nan;
        for j=1:10 % for each cluster-mean, find the closest training mean
            new_J = norm(C_256x10(:,i) - means(:,j));
            if new_J < J
                J = new_J;
                label = j-1;
            end
        end
        class_kmeans(i) = label;
    end

    
    %label index cluster for each data point as the corresponding digit 
    test_class_kmeans = NaN(1,4649); 
    for i=1:4649 
        test_class_kmeans(i) = class_kmeans(idx(i));
    end
    
    runtime_plusplus(s) = toc;

    if s == 1
        bestClusterAssign = class_kmeans;
    end

    %confusion matrix 
    confusionMatrix_kmeans = confusionmat(testingLabels,test_class_kmeans);
    %accuracy: number correctly classified (diagonal) / total
    accuracy_percent_kmeans = (sum(diag(confusionMatrix_kmeans))/sum(confusionMatrix_kmeans,"ALL"))*100;
    
    accuracies_plusplus(s) = accuracy_percent_kmeans;

end   


    
accuracies_plusplus
runtime_plusplus




