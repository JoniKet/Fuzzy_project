
function [train_sample, test_sample] = stratified_sampling(data, class_col, sampling_percentage)

% ===================================================================================================================
% Stratified random sampling method:
%   The method is to ensure that the training and testing data sets have similar proportions of classes, 
%   the sampling is stratified by class.      
%   INPUT:
%         data      :- The data to be classified for training and testing samples
%         class_col :- The column number of the data that contains class labels 
%         sampling_percentage :- The specified percentage for train/test
%         sample split. 
%   OUTPUT:
%         train_sample:- Classified training sample
%         test_sample :- Classified testing sample
% ===================================================================================================================

n = size(data,1); % Length of the data set
train = 1; % Create train label value
test  = 2; % Create test label value

split_data = repmat(test,n, 1); % Initialize the segment including test label
distinct_strata = unique(data(:, class_col)); % Determine distinct strata
num_distinct_strata = size(distinct_strata, 1); % Count distinct strata

for stratum = 1:num_distinct_strata   % Go through all distinct strata 
    
    stratum_index     = find(data(:, class_col) == distinct_strata(stratum)); % Find indexes of stratum in the data
    num_stratum_index = length(stratum_index); % Determine the size of index vector
    make_random       = randperm(num_stratum_index); % Generate a disorder of 'num_stratum_index' items
    split_data(stratum_index(make_random(1:round(sampling_percentage * num_stratum_index)))) = train; % Assign appropriate number of units to Training group

end

train_sample = data(split_data == train,:); % Train sample
test_sample  = data(split_data == test,:);  % Test sample

end

