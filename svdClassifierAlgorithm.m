%% Code for SVD Classification Algorithm

%% Reading in USPS Data

%%% Str Oject -- Can access each array via usps.[name]
usps = load('usps_resampled.mat')

train_patterns = usps.train_patterns
train_labels = usps.train_labels
test_patterns = usps.test_patterns
test_labels = usps.test_labels
%% View an image
img1vec = train_patterns(:,1)
img1mat = reshape(img1vec,[16,16])
imshow(img1mat') %tranpose or image flipped (??? idk why - says so in Saito's lec)

%% Obtain all training labels
trainingLabels = [];
for i = 1:4649
   val = find(train_labels(:,i)==1);
   trainingLabels(end+1) = val-1;
end

%%
%column indices for each digit
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
zeros = train_patterns(:,zeros_index);
ones = train_patterns(:,ones_index);
twos = train_patterns(:,twos_index);
threes = train_patterns(:,threes_index);
fours = train_patterns(:,fours_index);
fives = train_patterns(:,fives_index);
sixs = train_patterns(:,sixs_index);
sevens = train_patterns(:,sevens_index);
eights = train_patterns(:,eigths_index);
nines = train_patterns(:,nines_index);

%% SVD Algorithm

%%% Using k largest left singular vectors as approximate bases (Training)
k = 14
[U_0 S_0 V_0] = svds(zeros, k, 'largest')
[U_1 S_1 V_1] = svds(ones, k, "largest")
[U_2 S_2 V_2] = svds(twos, k, "largest")
[U_3 S_3 V_3] = svds(threes, k, 'largest')
[U_4 S_4 V_4] = svds(fours, k, 'largest')
[U_5 S_5 V_5] = svds(fives, k, 'largest')
[U_6 S_6 V_6] = svds(sixs, k, "largest")
[U_7 S_7 V_7] = svds(sevens, k, "largest")
[U_8 S_8 V_8] = svds(eights, k, 'largest')
[U_9 S_9 V_9] = svds(nines, k, 'largest')

%% X rank approximations of each digit

zerosApprox = U_0*S_0*V_0'
onesApprox = U_1*S_1*V_1'
twosApprox = U_2*S_2*V_2'
threesApprox = U_3*S_3*V_3'
foursApprox = U_4*S_4*V_4'
fivesApprox = U_5*S_5*V_5'
sixsApprox = U_6*S_6*V_6'
sevensApprox = U_7*S_7*V_7'
eightsApprox = U_8*S_8*V_8'
ninesApprox = U_9*S_9*V_9'

%%% See how bases performed visually:
zerovecApprox = zerosApprox(:, 1)
zeromatApprox = reshape(zerovecApprox, [16, 16])
imshow(zeromatApprox')

%% Classification (for single vector)

v = eights(:, 10) % Test Vector of handwritten digit
res0 = norm(v-U_0*U_0'*v, 2)
res1 = norm(v-U_1*U_1'*v, 2)
res2 = norm(v-U_2*U_2'*v, 2)
res3 = norm(v-U_3*U_3'*v, 2)
res4 = norm(v-U_4*U_4'*v, 2)
res5 = norm(v-U_5*U_5'*v, 2)
res6 = norm(v-U_6*U_6'*v, 2)
res7 = norm(v-U_7*U_7'*v, 2)
res8 = norm(v-U_8*U_8'*v, 2)
res9 = norm(v-U_9*U_9'*v, 2)

residuals = [res0 res2 res3 res4 res5 res6 res7 res8 res9]
if min(residuals) == res0
    classif = 0
elseif min(residuals) == res1
    classif = 1
elseif min(residuals) == res2
    classif = 2
elseif min(residuals) == res3
    classif = 3
elseif min(residuals) == res4
    classif = 4
elseif min(residuals) == res5
    classif = 5
elseif min(residuals) == res6
    classif = 6
elseif min(residuals) == res7
    classif = 7
elseif min(residuals) == res8
    classif = 8
elseif min(residuals) == res9
    classif = 9
end
disp(classif)

%% Using all of test data, and creating Confusion Matrix
% Finding Testing Indeces for Comparison/Validation
testingLabels = [];
for i = 1:4649
   val = find(test_labels(:,i)==1);
   testingLabels(end+1) = val-1;
end

zeros_index_test = find(testingLabels == 0);
ones_index_test = find(testingLabels == 1);
twos_index_test = find(testingLabels == 2);
threes_index_test = find(testingLabels == 3);
fours_index_test = find(testingLabels == 4);
fives_index_test = find(testingLabels == 5);
sixs_index_test = find(testingLabels == 6);
sevens_index_test = find(testingLabels == 7);
eigths_index_test = find(testingLabels == 8);
nines_index_test = find(testingLabels == 9);

zeros_test = test_patterns(:,zeros_index_test);
ones_test = test_patterns(:,ones_index_test);
twos_test = test_patterns(:,twos_index_test);
threes_test = test_patterns(:,threes_index_test);
fours_test = test_patterns(:,fours_index_test);
fives_test = test_patterns(:,fives_index_test);
sixs_test = test_patterns(:,sixs_index_test);
sevens_test = test_patterns(:,sevens_index_test);
eights_test = test_patterns(:,eigths_index_test);
nines_test = test_patterns(:,nines_index_test);

testMat = [zeros_test ones_test twos_test threes_test fours_test fives_test sixs_test sevens_test eights_test nines_test]
trueClass = [repelem(0, size(zeros_test, 2)) repelem(1, size(ones_test, 2)) repelem(2, size(twos_test, 2)) repelem(3, size(threes_test, 2)) repelem(4, size(fours_test, 2)) repelem(5, size(fives_test, 2)) repelem(6, size(sixs_test, 2)) repelem(7, size(sevens_test, 2)) repelem(8, size(eights_test, 2)) repelem(9, size(nines_test, 2))]

predClass = []
for i=1:size(testMat, 2)
    v = testMat(:, i)
    res0 = norm(v-U_0*U_0'*v, 2)
    res1 = norm(v-U_1*U_1'*v, 2)
    res2 = norm(v-U_2*U_2'*v, 2)
    res3 = norm(v-U_3*U_3'*v, 2)
    res4 = norm(v-U_4*U_4'*v, 2)
    res5 = norm(v-U_5*U_5'*v, 2)
    res6 = norm(v-U_6*U_6'*v, 2)
    res7 = norm(v-U_7*U_7'*v, 2)
    res8 = norm(v-U_8*U_8'*v, 2)
    res9 = norm(v-U_9*U_9'*v, 2)

    residuals = [res0 res2 res3 res4 res5 res6 res7 res8 res9]

    if min(residuals) == res1
        classif = 1
    elseif min(residuals) == res0
        classif = 0
    elseif min(residuals) == res2
        classif = 2
    elseif min(residuals) == res3
        classif = 3
    elseif min(residuals) == res4
        classif = 4
    elseif min(residuals) == res5
        classif = 5
    elseif min(residuals) == res6
        classif = 6
    elseif min(residuals) == res7
        classif = 7
    elseif min(residuals) == res8
        classif = 8
    elseif min(residuals) == res9
        classif = 9
    end
    predClass(i) = classif
end

confusionchart(trueClass, predClass)

