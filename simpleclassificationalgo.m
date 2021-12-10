%16x16 images, 9298 images total 

%test_labels 10x4649 (1 if number, -1 else)
%test_patterns 256*4649

%train_labels 10x4649 (1 if number, -1 else)
%train_patterns 256*4649

%view an image
% img1vec = train_patterns(:,1)
% img1mat = reshape(img1vec,[16,16])
% imshow(img1mat') %tranpose or image flipped (??? idk why - says so in Saito's lec)



load('usps_resampled.mat');

%obtain all testing labels
testingLabels = [];
for i = 1:4649
    val = find(test_labels(:,i)==1);
    testingLabels(end+1) = val-1; %digits 0-9
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

%%
%euclidean distance

%store classification as a 1x4649 array
test_class_l2 = NaN(1,4649);  
tic %calculcate run time 
%loop through every testing digit image
for i=1:4649 
    %initialize large distance between matricies
    J = Inf; 
    label = nan;
    %for each handwritten digit, find the closest mean
    for j=1:10 
        %euclidean distance
        new_J = norm(test_patterns(:,i) - means(:,j)); 
        if new_J < J
            J = new_J;
            label = j-1;
        end
    end
    %classify handwritten digit
    test_class_l2(i) = label;
end 
toc

%confusion matrix 
confusionMatrix_l2 = confusionmat(testingLabels,test_class_l2);

%accuracy: number correctly classified (diagonal) / total
accuracy_percent_l2 = (sum(diag(confusionMatrix_l2))/sum(confusionMatrix_l2,"ALL"))*100;

confusionchart(testingLabels,test_class_l2);

%%
%cosine distance

%store classification as a 1x4649 array
test_class_cos = NaN(1,4649);  
tic %calculcate run time 
%loop through every testing digit image
for i=1:4649 
    %initialize large distance between matricies
    J = Inf; 
    label = nan;
    %for each handwritten digit, find the closest mean
    for j=1:10 
        %cosine distance
        a = test_patterns(:,i);
        b = means(:,j);
        new_J = 1 - (dot(a,b)/(norm(a)*norm(b))); 
        if new_J < J
            J = new_J;
            label = j-1;
        end
    end
    %classify handwritten digit
    test_class_cos(i) = label;
end 
toc

%confusion matrix 
confusionMatrix_cos = confusionmat(testingLabels,test_class_cos);

%accuracy: number correctly classified (diagonal) / total
accuracy_percent_cos = (sum(diag(confusionMatrix_cos))/sum(confusionMatrix_cos,"ALL"))*100;


%% View means of digits
mean0 = reshape(means(:,1),[16,16]); %good
imshow(mean0');

mean1 = reshape(means(:,2),[16,16]); %good
imshow(mean1');

mean2 = reshape(means(:,3),[16,16]); %really bad
imshow(mean2');

mean3 = reshape(means(:,4),[16,16]); %good
imshow(mean3');

mean4 = reshape(means(:,5),[16,16]); %really bad
imshow(mean4');

mean5 = reshape(means(:,6),[16,16]); %decent
imshow(mean5');

mean6 = reshape(means(:,7),[16,16]); %decent
imshow(mean6');

mean7 = reshape(means(:,8),[16,16]); %good
imshow(mean7');

mean8 = reshape(means(:,9),[16,16]); %decent
imshow(mean8');

mean9 = reshape(means(:,10),[16,16]); %looks like a 7
imshow(mean9');

