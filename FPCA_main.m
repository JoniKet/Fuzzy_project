% Fuzzy principal component analysis for the practical assignment dataset
clear all; close all; clc;
data_train = readtable('data_train_S.csv');
data_test = readtable('data_test_S.csv');
xlabels = data_train.Properties.VariableNames(1:end-1);


data_train_table = table2array(data_train);
data_test_table = table2array(data_test);

Xtrain = data_train_table(:,1:end-1);
Ytrain = data_train_table(:,end)+1;
Xtest = data_test_table(:,1:end-1);
Ytest = data_test_table(:,end)+1;

%% PCA  AND FPCA

[PC,w]=frpca(data_train_table(:,1:end-1),47);
[t,p,r2] = pca(data_train_table(:,1:end-1));


Xtrain_F_PCA = PC(:,1:end);
Xtrain_PCA = t(:,1:end);

%xtrain = PC/w this gets original Xtrain

Xtest_F_PCA = Xtest*w;
Xtest_PCA = Xtest*p;

%% fuzzy k-nearest neigbours


% Using fuzzy PCA

[Y1,MEMS1,HITS1] = fknn(Xtrain_F_PCA, Ytrain, Xtest_F_PCA, Ytest, 1, 0,1); 
[acc,sen,spe] = accSenSpeCalc(Y1,Ytest);
fprintf('FUZZY PCA & FUZZY K-NN TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


% Using normal PCA with fuzzy K-nearest neighbours
[Y2,MEMS2,HITS2] = fknn(Xtrain_PCA, Ytrain, Xtest_PCA, Ytest,1:10, 0,1); 
[acc,sen,spe] = accSenSpeCalc(Y2,Ytest);
fprintf('NORMAL PCA & FUZZY K-NN TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


% using fuzzy k-nearest neighbours without PCA
[Y3,MEMS3,HITS3] = fknn(Xtrain, Ytrain, Xtest, Ytest,1:10, 0,1); 
[acc,sen,spe] = accSenSpeCalc(Y3,Ytest);
fprintf('NO PCA & FUZZY K-NN TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)

%% non fuzzy K-nn 

% WITHOUT PCA

mdl = fitcknn(Xtrain,Ytrain)
cvmdl = crossval(mdl)

[label,score,cost] = predict(mdl,Xtrain); % train set acc and stuff
[acc,sen,spe] = accSenSpeCalc(label,Ytrain);
fprintf('no Pca K-nn TRAIN SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)

[label,score,cost] = predict(mdl,Xtest); % test set acc and stuff
[acc,sen,spe] = accSenSpeCalc(label,Ytest);
fprintf('no Pca K-nn TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)

% USING PCA

mdl = fitcknn(Xtrain_PCA,Ytrain)
cvmdl = crossval(mdl);

[label,score,cost] = predict(mdl,Xtrain_PCA); % train set acc and stuff
[acc,sen,spe] = accSenSpeCalc(label,Ytrain);
fprintf('PCA K-nn TRAIN SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)

[label,score,cost] = predict(mdl,Xtest_PCA); % test set acc and stuff
[acc,sen,spe] = accSenSpeCalc(label,Ytest);
fprintf('PCA K-nn TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


% USING FUZZY PCA

mdl = fitcknn(Xtrain_F_PCA,Ytrain)
cvmdl = crossval(mdl);

[label,score,cost] = predict(mdl,Xtrain_F_PCA); % train set acc and stuff
[acc,sen,spe] = accSenSpeCalc(label,Ytrain);
fprintf('FPCA K-nn TRAIN SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)

[label,score,cost] = predict(mdl,Xtest_F_PCA); % test set acc and stuff
[acc,sen,spe] = accSenSpeCalc(label,Ytest);
fprintf('FPCA K-nn TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


%% Simple regression analysis

% Using only normalized data

%training set
mdl = fitlm(Xtrain,Ytrain);
[label,score] = predict(mdl,Xtrain); % train set acc and stuff
for i = 1:length(label)
  if label(i) >= 1.5
      label(i) = 2;
  elseif label(i) < 1.5
      label(i) = 1;
  end
end

[acc,sen,spe] = accSenSpeCalc(label,Ytrain);
fprintf('Simple linear regression (no pca) TRAIN SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)

% test set
[label,score] = predict(mdl,Xtest); % test set acc and stuff
for i = 1:length(label)
  if label(i) >= 1.5
      label(i) = 2;
  elseif label(i) < 1.5
      label(i) = 1;
  end
end
[acc,sen,spe] = accSenSpeCalc(label,Ytest);
fprintf('Simple linear regression (no pca) TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)



% Using PCA data

%training set
mdl = fitlm(Xtrain_PCA,Ytrain);
[label,score] = predict(mdl,Xtrain_PCA); % train set acc and stuff
for i = 1:length(label)
  if label(i) >= 1.5
      label(i) = 2;
  elseif label(i) < 1.5
      label(i) = 1;
  end
end

[acc,sen,spe] = accSenSpeCalc(label,Ytrain);
fprintf('Simple linear regression (PCA) Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


%test set
[label,score] = predict(mdl,Xtest_PCA); % train set acc and stuff
for i = 1:length(label)
  if label(i) >= 1.5
      label(i) = 2;
  elseif label(i) < 1.5
      label(i) = 1;
  end
end

[acc,sen,spe] = accSenSpeCalc(label,Ytest);
fprintf('Simple linear regression (PCA) Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


% Using FPCA xdata
%training set
mdl = fitlm(Xtrain_F_PCA,Ytrain);
[label,score] = predict(mdl,Xtrain_F_PCA); % train set acc and stuff
for i = 1:length(label)
  if label(i) >= 1.5
      label(i) = 2;
  elseif label(i) < 1.5
      label(i) = 1;
  end
end
[acc,sen,spe] = accSenSpeCalc(label,Ytrain);
fprintf('Simple linear regression (FPCA) Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


%test set
[label,score] = predict(mdl,Xtest_F_PCA); % train set acc and stuff
for i = 1:length(label)
  if label(i) >= 1.5
      label(i) = 2;
  elseif label(i) < 1.5
      label(i) = 1;
  end
end

[acc,sen,spe] = accSenSpeCalc(label,Ytest);
fprintf('Simple linear regression (FPCA) Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)

















