% Fuzzy data analysis practical assignment FKNN CROSS PCA

clear all
close all
clc;
warning off

data = readtable('data_whole.csv');
data_new = table2array(data);

%% WARNING 

% IT IS NOT GOOD PRACTISE to do PCA for the whole dataset. However due to
% limitations in computing power, the PCA is done once for the whole
% dataset instead of calculating weights every time in the cross validation
% loop. 

%% PCA

[t,p,r2] = pca(scale(data_new(:,1:end-1)));

train_data = t(1:round(length(t(:,1))*0.7),:); % dataset that is used for cross validation
train_data_y = data_new(1:round(length(t(:,1))*0.7),end);
test_data = t(length(train_data):end,:); % data that is used to test the model with optimal parameters
test_data_y = data_new(length(train_data):end,end);

figure(1)
title('R^2 graph')
hold on
plot(1:47,r2)
plot(1:47,r2,'o')
xlabel('Number of parameters included');
ylabel('Variability in the data explained');
grid on;
hold off




%% FKNN PARAMETERS
N=30; % How many times random division to training set and testing set is done
K=[1:5]; % numbers of k-nn to test
rn = 1/2; % amount of data in validation and split set
mink = 2; % minumum amount of variables to include
maxk = 47; % max amount of variables to include

%% FKNN loop


for k = mink:1:maxk

  data_new = [train_data(:,1:k) train_data_y]; % Defines how many variables to include
  [rows, columns] = size(data_new);

  % sorting the observations with the example script given in exercies. 
  [data, lc, cs]= init_data(data_new,1:columns-1,columns);

  
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
         train_labels, test,test_labels, K,0,1);
  %    results=numhits/length(test_labels);

     for l = 1:max(K) % choosing which k value produces highest sensitivity
        [acc(l),sen(l),spe(l)] = accSenSpeCalc(y(:,l),test_labels); 
     end
     accuracy = [accuracy; acc];
     sensitivity = [sensitivity; sen];
     specificity = [specificity; spe];
     fprintf('Check %2.0f out of %2.0f \n',i,N);

  end
  MeansACC(:,k) = mean(accuracy);    %Mean classification accuracies from 30 repetation rows = number of neighbouts columns = number of variables
  MeansSEN(:,k) = mean(sensitivity);
  MeansSPE(:,k) = mean(specificity);
  fprintf('Completion %2.0f out of %2.0f \n',k,maxk);
end

% getting rid of the zero columns
% 
MeansACC = MeansACC(:,mink:end);
MeansSEN = MeansSEN(:,mink:end);
MeansSPE = MeansSPE(:,mink:end);

%% Plotting and results evaluation


accArray = reshape(MeansACC,[],1);
senArray = reshape(MeansSEN,[],1);
speArray = reshape(MeansSPE,[],1);
varArray = []; kArray = [];
for i = 1:max(K)
  varArray = [varArray mink:maxk];
end
for i = 1:(maxk-mink+1)
  kArray = [kArray 1:max(K)];
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
fprintf('Model performance with optimal parameter (max sensitivity) TRAIN SET: \nACC: %2.4f \nSEN: %2.4f \nSPE: %2.4f \nK-nn neighbours included: %1.0f \nNum of columns from data: %1.0f \n' ,...
  accArray(idx),senArray(idx),speArray(idx),kArray(idx),varArray(idx))


%% Using optimal parameters in the test data

% in order to use the function +1 has to be added to each row of
% train_data_y and test_data_y

train_data_y = train_data_y +1;
test_data_y = test_data_y +1;

[y2,mem, numhits] = fknn(train_data(:,1:varArray(idx)),...
         train_data_y(:,1), test_data(:,1:varArray(idx)), test_data_y(:,1),kArray(idx),0,1);

       

[acc,sen,spe] = accSenSpeCalc(y2,test_data_y(:,1));

fprintf('Model performance with optimal parameter (max sensitivity) TEST SET: \nACC: %2.4f \nSEN: %2.4f \nSPE: %2.4f \nK-nn neighbours included: %1.0f \nNum of columns from data: %1.0f' ,...
  acc,sen,spe,kArray(idx),varArray(idx));









