% Similiarity classifier fuzzy data analysis cource practical assignment

clear all
close all
clc;
warning off

%% Loading the data

data = readtable('data_whole.csv');
data_new = table2array(data);

%% Classification with similarity

v = [1:47]; % columns of independent variables
c = 48;  % column of dependent variable
splitvalue=0.7;
[train_sample, test_sample] = stratified_sampling(data_new, c, splitvalue);

measure = 1; % Used similarity measure (see classifier.m) 
p = [0.1:0.25:4]; % p parameter range
m = [0.25:0.25:3]; % m parameter range 
pl=1; % Do we use plotting w.r.t. parameters p and m or not pl=0 no plotting pl=1 plotting.
N=100; %How many times data is divided into validation and training sets.
rn=1/2; % Now data is divided in half; half for the validation set and half for the training set.

[Mean_accuracy, Variance,p1,m1,pp,mm,id,maxsen,Mean_sensitivity,Mean_specificity,Mean_FPR,Mean_FNR]=simclass4(train_sample,v,c, measure, p, m, N,rn,pl);

%%
y = [pp,mm,measure]; % p and m values and similarity measure      
y2 = [p(p1),m(m1),measure];



fprintf('Model performance with optimal parameter (max sensitivity) TRAIN SET: \nACC: %2.4f \nSEN: %2.4f \nSPE: %2.4f \nP parameter: %1.2f \nM parameter: %1.2f \n' ,...
  Mean_accuracy,Mean_sensitivity,Mean_specificity,pp,mm)

%% Test set results

%As an output we now get mean accuracies and variances from validation set
%with best parameter p and m values. Besides this also we take out ideal
%vectors, p and m value (id,pp,mm) for which we got the best results in the
%validation set (or first one of those in case of several).
%With these parameters (id,pp,mm) we still run results using testing set as
%a last step:

[datatest, lc, cs] = init_data(test_sample,v,c); % data initialization
[accuracy, class, Simil] = calcfitness2(datatest, id, y);
[acc,sen,spe] = accSenSpeCalc(class,datatest(:,c));
fprintf('Model performance with optimal parameter (max sensitivity) TEST SET: \nACC: %2.4f \nSEN: %2.4f \nSPE: %2.4f \nP parameter: %1.2f \nM parameter: %1.2f \n' ,...
  acc,sen,spe,pp,mm)


[accuracy2, class2, Simil2] = calcfitness2(datatest, id, y2);
[acc2,sen2,spe2] = accSenSpeCalc(class2,datatest(:,c));
fprintf('Model performance with optimal parameter (max sensitivity) TEST SET \n WITH OPTIMAL PARAMETERS ON AVERAGE: \nACC: %2.4f \nSEN: %2.4f \nSPE: %2.4f \nP parameter: %1.2f \nM parameter: %1.2f \n' ,...
  acc2,sen2,spe2,p(p1),m(m1))


% 
% [TP,FP,FN,TN]=performancemeasures(class,datatest(:,c)) ;







