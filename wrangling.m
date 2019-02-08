clc; clear all; close all;

%Practical assignment data wrangling.

data = readtable('bank.csv');
data_new = data;


%% Removing NaN and imputating




%% CATEGORICAL VARIABLES

% The binary variables can be left alone, since there are only two
% categories.

% Some of the variables have to be transformed from categorical form to
% numerical form. These variables contain ordinal data meaning the order of
% the alternatives is known but the distance of the alternatives is not
% known

% These variables are ordinal: education, 
data.Edu=categorical(data.education);

for i=1:length(data.Edu)
    if isequal(data.Edu(i),'primary')
        data_new.Edu(i) = 1;
    elseif isequal(data.Edu(i),'secondary')
        data_new.Edu(i) = 2;
    elseif isequal(data.Edu(i),'tertiary')
        data_new.Edu(i) = 3;
    elseif isequal(data.Edu(i),'unknown')
        data_new.Edu(i) = NaN;
    end
end

data_new = removevars(data_new,'education');
data = removevars(data,'education');

% These variables have categories, but no clear ranking between the
% variables: job, marital, contact, last contact month, poutcome

% JOB
data.job=categorical(data.job);
X = dummyvar(data.job);
jobs = categories(data.job); 

for i = 1:length(jobs)
  if isequal(jobs{i},'admin.')
    jobs{i} = 'admin';
  elseif isequal(jobs{i},'blue-collar')
    jobs{i} = 'blueCollar';
  elseif isequal(jobs{i},'self-employed')
    jobs{i} = 'selfEmployed';
  end
end

jobs_dummies = array2table(X);
jobs_dummies.Properties.VariableNames = jobs;
data_new = addvars(data_new,jobs_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for job 
data_new = removevars(data_new,'job');


% Marital

maritals_dummies = to_categorical(data.marital);
data_new = addvars(data_new,maritals_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for marital status
data_new = removevars(data_new,'marital');

% contact

data.contact=categorical(data.contact);
X = dummyvar(data.contact);
contacts = categories(data.contact); 
contacts_dummies = array2table(X);
contacts_dummies.Properties.VariableNames = contacts;
data_new = addvars(data_new,contacts_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for contact method
data_new = removevars(data_new,'contact');

% last contact month

data.month=categorical(data.month);
X = dummyvar(data.month);
months = categories(data.month); 
months_dummies = array2table(X);
months_dummies.Properties.VariableNames = months;
data_new = addvars(data_new,months_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for contact month
data_new = removevars(data_new,'month');

% poutcome

data.poutcome=categorical(data.poutcome);
X = dummyvar(data.poutcome);
poutcomes = categories(data.poutcome); 
poutcomes_dummies = array2table(X);
poutcomes_dummies.Properties.VariableNames = poutcomes;
data_new = addvars(data_new,poutcomes_dummies,'After',data_new.Properties.VariableNames(end)); % now the dataset has dummy for outcome of previous marketing
data_new = removevars(data_new,'poutcome');


% REMOVING THE 'UNKNOWN' DUMMY COLUMNS





% Transforming binary 'yes' 'no' variables to 1 0 ------------------



%% normalizing the variables






