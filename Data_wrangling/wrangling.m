clc; clear all; close all;

%Practical assignment data wrangling.

data = readtable('bank-full.csv');
data_new = data;





%% CATEGORICAL VARIABLES

% This part takes care of missing or unknown categorical values by first
% categorizing and then removing the unknown column. 

% Some of the variables have to be transformed from categorical form to
% numerical form. These variables contain ordinal data meaning the order of
% the alternatives is known but the distance of the alternatives is not
% known
data_new.education=categorical(data.education);
% These variables are ordinal: education, 

for i=1:length(data_new.education)
    if isequal(data_new.education(i),'primary')
        data_new.Edu(i) = 1;
    elseif isequal(data_new.education(i),'secondary')
        data_new.Edu(i) = 2;
    elseif isequal(data_new.education(i),'tertiary')
        data_new.Edu(i) = 3;
    elseif isequal(data_new.education(i),'unknown')
        data_new.Edu(i) = NaN;
    end
end

data_new = removevars(data_new,'education');

% These variables have categories, but no clear ranking between the
% variables: job, marital, contact, last contact month, poutcome

% JOB | since jobs has variable names which are not compatible with matlab,
% names have to be reprocessed

temp=categorical(data_new.job);
jobs = categories(temp); 
for i = 1:length(jobs)
  if isequal(jobs{i},'admin.')
    jobs{i} = 'admin';
  elseif isequal(jobs{i},'blue-collar')
    jobs{i} = 'blueCollar';
  elseif isequal(jobs{i},'self-employed')
    jobs{i} = 'selfEmployed';
  end
end

jobs_dummies = to_categorical(data_new.job,jobs,0,'jobUnknown');
data_new = addvars(data_new,jobs_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for job 
data_new = removevars(data_new,'job');


% Marital

maritals_dummies = to_categorical(data_new.marital,[],1,[]);
data_new = addvars(data_new,maritals_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for marital status
data_new = removevars(data_new,'marital');

% contact

contacts_dummies = to_categorical(data_new.contact,[],1,[]);
data_new = addvars(data_new,contacts_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for contact method
data_new = removevars(data_new,'contact');

% last contact month

months_dummies = to_categorical(data_new.month,[],1,[]);
data_new = addvars(data_new,months_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for contact month
data_new = removevars(data_new,'month');

% poutcome

poutcomes_dummies = to_categorical(data_new.poutcome,[],1,[]);
data_new = addvars(data_new,poutcomes_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for outcome of previous marketing
data_new = removevars(data_new,'poutcome');


% Transforming binary 'yes' 'no' variables to 1 0 ------------------

binvars = {'default','housing','loan','y'};
for i = 1:length(binvars)
  var = binvars{i};
  array = zeros(length(data_new{:,var}),1);
  IndexNo = find(contains(data_new{:,var},'no'));
  IndexYes = find(contains(data_new{:,var},'yes'));
  array(IndexNo) = 0;
  array(IndexYes) = 1;
  data_new = removevars(data_new,var);
  if isequal(var,'y')
    data_new = addvars(data_new,array,'after',data_new.Properties.VariableNames(end),'NewVariableName',var);
  else
    data_new = addvars(data_new,array,'before','age','NewVariableName',var);
  end
end

%% handling the pdays variable

%Since pdays contains 'categorical values' aka -1 and other number values,
%it has to be transformed somehow. Fuzzy trapezoidal numbers are used to
%define how long the time that passed by after the client was contacted
%from a previous campaign
histogram(data.pdays)
neverContactedF = [-100 -100 -1 -0.1]; % Taking the -1 never contacted values to this crisp set
recentlyContactedF = [0 50 100 150];
sometimeAgoF = [100 150 200 250];
aWhileAgoF = [200 250 300 350];
longTimeAgoF = [300 350 1000 1000];

nC = zeros(length(data_new.pdays),1); rC = zeros(length(data_new.pdays),1); sometimeAgo = zeros(length(data_new.pdays),1); aWhileAgo = zeros(length(data_new.pdays),1); 
longTimeAgo = zeros(length(data_new.pdays),1);

for i = 1:length(data.pdays)
  nC(i) = trapf2(data.pdays(i),neverContactedF);
  rC(i) = trapf2(data.pdays(i),recentlyContactedF);
  sometimeAgo(i) = trapf2(data.pdays(i),sometimeAgoF);
  aWhileAgo(i) = trapf2(data.pdays(i),aWhileAgoF);
  longTimeAgo(i) = trapf2(data.pdays(i),longTimeAgoF);
end

%saving the membership values to the dataset
data_new = addvars(data_new,nC,rC,sometimeAgo,aWhileAgo,longTimeAgo,'before','age','NewVariableName',{'neverContacted','recentlyContacted' ...
  ,'sometimeAgoContacted','aWhileAgoContacted','longTimeAgoContacted'},'before','y');

data_new = removevars(data_new,'pdays');
%% Removing NaN and imputating

data_new = splitvars(data_new);
temp = table2array(data_new);
temp2 = knnimpute(temp);
data_new = array2table(temp2,'VariableNames',data_new.Properties.VariableNames);


%% Normalization of the parameters using min max

% data_new.age = minmaxnorm(data_new.age,0,1);
% data_new.balance = minmaxnorm(data_new.balance,0,1);
% data_new.day = minmaxnorm(data_new.day,0,1);
% data_new.duration = minmaxnorm(data_new.duration,0,1);
% data_new.previous = minmaxnorm(data_new.previous,0,1);
% data_new.campaign = minmaxnorm(data_new.campaign,0,1);


%% Dividing data into train and test set

trainlength = round(length(data_new.y)*0.7);
data_train = data_new(1:trainlength,:);
data_test = data_new(trainlength:end,:);

% saving the data_train and data_test
if length(data_new.y) < 5000
  writetable(data_train,'data_train_S.csv') %smaller dataset
  writetable(data_test,'data_test_S.csv')
else
  writetable(data_train,'data_train.csv') %bigger dataset
  writetable(data_test,'data_test.csv')
end



% Saving also the whole dataset for cross validation purposes
if length(data_new.y) < 5000
  writetable(data_new,'data_whole_S.csv')
else
  writetable(data_new,'data_whole.csv')
end




% EOF

