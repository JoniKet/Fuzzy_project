% Fuzzy data analysis FKNN with cross validation

clear all
close all
clc;
warning off

data = readtable('data_whole_S.csv');
data_new = table2array(data);
[rows, columns] = size(data_new);

% sorting the observations with the example script given in exercies. 
[data, lc, cs]= init_data(data_new,1:columns-1,columns);

rn=1/2; %Amount of data to the training set

N=100; % How many times random division to training set and testing set is done
v=columns-1;
c=columns;

accuracy=[]; %Store the classification accuracies
sensitivity = [];
specificity = [];
%Implemented with crossvalidation:

rn_train=ones(1,length(lc))*rn(1);

for i=1:N %Randomly repeat division to training set and testing set
   train_ind=[];
   for j=1:length(lc) %For each class
       temp=randperm(lc(j))-1; %random permutation of the class sample indexes
       %Indexes of the training set:
       train_ind=[train_ind, cs(j)+temp(1:floor(lc(j)*rn_train(j)))];
       %Indexes of the testing set:
   end
   test_ind=setxor([1:size(data,1)],train_ind); %All the indexes which are not in training set
   %are selected to testing set.
   test=data(test_ind,1:v); %data for verification
   train=data(train_ind,1:v); %data for training
   test_labels=data(test_ind,c); %Class labels for testing set
   train_labels=data(train_ind,c); %Class labels for training set
   K=[1:10];
   [y,mem, numhits] = fknn(train,...
       train_labels, test,test_labels, K,0,1);
%    results=numhits/length(test_labels);
   
   for i = 1:10 % choosing which k value produces highest sensitivity
      [acc(i),sen(i),spe(i)] = accSenSpeCalc(y(:,i),test_labels); 
   end
   accuracy = [accuracy; acc];
   sensitivity = [sensitivity; sen];
   specificity = [specificity; spe];

end
MeansACC=mean(accuracy);    %Mean classification accuracies from 30 repetation
MeansSEN = mean(sensitivity);
MeansSPE = mean(specificity);
Vars=var(accuracy);      %Variances



figure(1)
subplot(311)
hold on
title('Mean accuracy w.r.t. no of nearest neighbors','FontSize',15)
plot(MeansACC,'LineWidth',2)
xlabel('K nearest neighbors','FontSize',15)
ylabel('Mean accuracy','FontSize',15)
hold off

subplot(312)
hold on
title('Mean sensitivity w.r.t. no of nearest neighbors','FontSize',15)
plot(MeansSEN,'LineWidth',2)
xlabel('K nearest neighbors','FontSize',15)
ylabel('Mean sensitivity','FontSize',15)
hold off

subplot(313)
hold on
title('Mean specificity w.r.t. no of nearest neighbors','FontSize',15)
plot(MeansSPE,'LineWidth',2)
xlabel('K nearest neighbors','FontSize',15)
ylabel('Mean specificity','FontSize',15)
hold off








