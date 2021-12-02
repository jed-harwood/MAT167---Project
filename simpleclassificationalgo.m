load('usps_resampled.mat');

%16x16 images, 9298 images total 

%test_labels 10x4649 (1 if number, -1 else)
%test_patterns 256*4649

%train_labels 10x4649 (1 if number, -1 else)
%train_patterns 256*4649

%view an image
% img1vec = train_patterns(:,1)
% img1mat = reshape(img1vec,[16,16])
% imshow(img1mat') %tranpose or image flipped (??? idk why - says so in Saito's lec)


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

test_class = NaN(1,4649); %store classification should be 1x4649
for i=1:4649
    J = Inf; %initialize large distance between matricies
    label = nan;
    for j=1:10 %for each handwritten digit, find the closest mean
        new_J = norm(test_patterns(:,i) - means(:,j)); 
        if new_J < J
            J = new_J;
            label = j-1;
        end
    end
    %classify handwritten digit
    test_class(i) = label;
end 

%confusion matrix 
confusionMatrix = confusionmat(testingLabels,test_class);

accuracy_percent = (sum(diag(confusionMatrix))/sum(confusionMatrix,"ALL"))*100;
