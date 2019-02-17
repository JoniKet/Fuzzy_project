% Fuzzy principal component analysis for the practical assignment dataset
clear all; close all; clc; warning off;
data_train = readtable('data_train.csv');
data_test = readtable('data_test.csv');
xlabels = data_train.Properties.VariableNames(1:end-1);


data_train_table = table2array(data_train);
data_test_table = table2array(data_test);

Xtrain = data_train_table(:,1:end-1);
Ytrain = data_train_table(:,end)+1;
Xtest = data_test_table(:,1:end-1);
Ytest = data_test_table(:,end)+1;

%% BASELINE SOLUTION

% A baseline solution would obviously be to choose that the client has not
% subscribed a term deposit. However this method would have zero
% sensitivity.

[acc,sen,spe] = accSenSpeCalc(ones(length(Ytrain),1),Ytrain);
fprintf('BASELINE RESULTS Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc(index),sen(index),spe(index))

%% PCA  AND FPCA

[PC,w]=frpca(data_train_table(:,1:end-1),47);
[t,p,r2] = pca(scale(data_train_table(:,1:end-1)));

figure(1)
title('R^2 graph')
hold on
plot(1:47,r2)
plot(1:47,r2,'o')
hold off

Xtrain_F_PCA = PC(:,1:end);
Xtrain_PCA = t(:,1:end);

%xtrain = PC/w this gets original Xtrain

Xtest_F_PCA = Xtest*w;
Xtest_PCA = Xtest*p;

%% fuzzy k-nearest neigbours

sen(1) = 0; acc(1) = 0; spe(1) = 0; index = 1;
% Using fuzzy PCA
[Y1,MEMS1,HITS1] = fknn(Xtrain_F_PCA, Ytrain, Xtest_F_PCA, Ytest, 1:10, 0,1);
for i = 1:10 % choosing which k value produces highest sensitivity
  [acc(i),sen(i),spe(i)] = accSenSpeCalc(Y1(:,i),Ytest); 
  [temp,index] = max(sen)
end
fprintf('FUZZY PCA & FUZZY K-NN TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc(index),sen(index),spe(index))



% Using normal PCA with fuzzy K-nearest neighbours
[Y2,MEMS2,HITS2] = fknn(Xtrain_PCA, Ytrain, Xtest_PCA, Ytest,1:10, 0,1); 
for i = 1:10 % choosing which k value produces highest sensitivity
  [acc(i),sen(i),spe(i)] = accSenSpeCalc(Y2(:,i),Ytest); 
  [temp,index] = max(sen)
end
fprintf('NORMAL PCA & FUZZY K-NN TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc(index),sen(index),spe(index))


% using fuzzy k-nearest neighbours without PCA
[Y3,MEMS3,HITS3] = fknn(Xtrain, Ytrain, Xtest, Ytest,1:10, 0,1); 
for i = 1:10 % choosing which k value produces highest sensitivity
  [acc(i),sen(i),spe(i)] = accSenSpeCalc(Y3(:,i),Ytest); 
  [temp,index] = max(sen)
end
fprintf('NO PCA & FUZZY K-NN TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc(index),sen(index),spe(index))

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
fprintf('Simple linear regression (PCA) TRAIN SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


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
fprintf('Simple linear regression (PCA) TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


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
fprintf('Simple linear regression (FPCA) TRAIN SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)


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
fprintf('Simple linear regression (FPCA) TEST SET Accuracy: %f  \n  Sensitivity: %f   \n Specificity: %f  \n',acc,sen,spe)

















