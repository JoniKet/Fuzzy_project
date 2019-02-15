% Fuzzy data analysis practical assignment

clear all
close all
clc;
warning off

data = readtable('data_whole.csv');
data_new = table2array(data);


%% WARNING 

% IT IS NOT GOOD PRACTISE to do FPCA for the whole dataset. However due to
% limitations in computing power, the FPCA is done once for the whole
% dataset instead of calculating weights every time in the cross validation
% loop. 

%% NO pca in this method
% [PC,w]=frpca(data_new(:,1:end-1),47);





%% FKNN
N=50; % How many times random division to training set and testing set is done
K=[1:5]; % numbers of k-nn to test
MeansACC = zeros(max(K),20); MeansSEN = zeros(max(K),20); MeansSPE = zeros(max(K),20);
for k = 20:1:39

  data_new = [data_new(:,1:k) data_new(:,end)]; % Defines how many variables to include
  [rows, columns] = size(data_new);

  % sorting the observations with the example script given in exercies. 
  [data, lc, cs]= init_data(data_new,1:columns-1,columns);

  rn=2/4; %Amount of data to the training set

  
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
     [y,mem, numhits] = fknn(train,...
         train_labels, test,test_labels, K,0,'false');
  %    results=numhits/length(test_labels);

     for i = 1:max(K) % choosing which k value produces highest sensitivity
        [acc(i),sen(i),spe(i)] = accSenSpeCalc(y(:,i),test_labels); 
     end
     accuracy = [accuracy; acc];
     sensitivity = [sensitivity; sen];
     specificity = [specificity; spe];

  end
  MeansACC(:,k)=mean(accuracy);    %Mean classification accuracies from 30 repetation rows = number of neighbouts columns = number of variables
  MeansSEN(:,k) = mean(sensitivity);
  MeansSPE(:,k) = mean(specificity);
end

% getting rid of the zero columns
% 
MeansACC = MeansACC(:,20:end);
MeansSEN = MeansSEN(:,20:end);
MeansSPE = MeansSPE(:,20:end);

%% Plotting and results evaluation


accArray = reshape(MeansACC,[],1);
senArray = reshape(MeansSEN,[],1);
speArray = reshape(MeansSPE,[],1);
varArray = []; kArray = [];
for i = 1:5
  varArray = [varArray 20:39];
end
for i = 1:20
  kArray = [kArray 1:5];
end

% plotting ACC
figure(2)
hold on
scatter3(kArray,varArray,accArray,'r');
[xq,yq] = meshgrid(min(kArray):.01:max(kArray), min(varArray):.01:max(varArray));
vq = griddata(kArray,varArray,accArray,xq,yq);
mesh(xq,yq,vq);
xlabel('Number of k nearest neighbours in algo'); ylabel('Number of variables included from the data'); zlabel('Classification model Accuracy');
title('Accuracy w.r.t amount of k-nn neighbors and variables from the data')
grid on;
hold off
% plotting SEN
figure(3)
hold on
scatter3(kArray,varArray,senArray,'r')
[xq,yq] = meshgrid(min(kArray):.01:max(kArray), min(varArray):.01:max(varArray));
vq = griddata(kArray,varArray,senArray,xq,yq);
mesh(xq,yq,vq)
xlabel('Number of k nearest neighbours in algo'); ylabel('Number of variables included from the data'); zlabel('Classification model Sensitivity');
title('Sensitivity w.r.t amount of k-nn neighbors and variables from the data')
grid on;
hold off

% plotting SPE
figure(4)
hold on
scatter3(kArray,varArray,speArray,'r');
[xq,yq] = meshgrid(min(kArray):.01:max(kArray), min(varArray):.01:max(varArray));
vq = griddata(kArray,varArray,speArray,xq,yq);
mesh(xq,yq,vq);
xlabel('Number of k nearest neighbours in algo'); ylabel('Number of variables included from the data'); zlabel('Classification model Specificity');
title('Specificity w.r.t amount of k-nn neighbors and variables from the data')
grid on;
hold off

[value,idx] = max(senArray);
fprintf('When sensitivity is at max: \nACC: %2.4f \nSEN: %2.4f \nSPE: %2.4f \nK-nn neighbours included: %1.0f \nNum of columns from data: %1.0f' ,...
  accArray(idx),senArray(idx),speArray(idx),kArray(idx),varArray(idx))