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

jobs_dummies = to_categorical(data_new.job,jobs);
data_new = addvars(data_new,jobs_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for job 
data_new = removevars(data_new,'job');


% Marital

maritals_dummies = to_categorical(data_new.marital);
data_new = addvars(data_new,maritals_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for marital status
data_new = removevars(data_new,'marital');

% contact

contacts_dummies = to_categorical(data_new.contact);
data_new = addvars(data_new,contacts_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for contact method
data_new = removevars(data_new,'contact');

% last contact month

months_dummies = to_categorical(data_new.month);
data_new = addvars(data_new,months_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for contact month
data_new = removevars(data_new,'month');

% poutcome

poutcomes_dummies = to_categorical(data_new.poutcome);
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




%% Removing NaN and imputating

data_new = splitvars(data_new);
temp = table2array(data_new);
temp2 = knnimpute(temp);
data_new = array2table(temp2,'VariableNames',data_new.Properties.VariableNames);

%% normalizing the variables






