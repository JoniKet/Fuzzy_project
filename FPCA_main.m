% Fuzzy principal component analysis for the practical assignment dataset
clear all; close all; clc;
data_train = readtable('data_train_S.csv');
data_test = readtable('data_test_S.csv');
xlabels = data_train.Properties.VariableNames(1:end-1);


data_train_table = table2array(data_train);
data_test_table = table2array(data_test);

Ytrain = data_train_table(:,end)+1;
Xtest = data_test_table(:,1:end-1);
Ytest = data_test_table(:,end)+1;


% Using fuzzy PCA
[PC,w]=frpca(data_train_table(:,1:end-1),47);

Xtrain = PC(:,1:end);
[Y1,MEMS1,HITS1] = fknn(Xtrain, Ytrain, Xtest, Ytest, 5, 0); 

fprintf('fuzzy-PCA and Fuzzy k-nearest neigbours accuracy in test set: %f  \n',HITS1/length(Ytest)) %fpca was able to detect outliers!


% Using normal PCA with fuzzy K-nearest neighbours
[t,p,r2] = pca(data_train_table(:,1:end-1));

Xtrain2 = t(:,1:end);

[Y2,MEMS2,HITS2] = fknn(Xtrain2, Ytrain, Xtest, Ytest, 5, 0); 

fprintf('Normal PCA and fuzzy K-nn accuracy in test set: %f  \n',HITS2/length(Ytest))


% using fuzzy k-nearest neighbours without PCA
Xtrain3 = data_train_table(:,1:end-1);

[Y3,MEMS3,HITS3] = fknn(Xtrain3, Ytrain, Xtest, Ytest, 5, 0); 

fprintf('Scaled data and fuzzy K-nn accuracy in test set: %f  \n',HITS3/length(Ytest))

%% non fuzzy K-nn without PCA
Xtrain4 = data_train_table(:,1:end-1);

mdl = fitcknn(Xtrain4,Ytrain,'NumNeighbors',5)
cvmdl = crossval(mdl)
[label,score,cost] = predict(mdl,Xtrain4);




























