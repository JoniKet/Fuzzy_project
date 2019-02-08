function new_data = to_categorical(variable, names)

  variable = categorical(variable);
  X = dummyvar(variable);
  
  if nargin < 2
    names = categories(variable);
  end
  
  dummy_table = array2table(X);
  dummy_table.Properties.VariableNames = names;
  
  
  if any(strcmp(names,'unknown')) == 1
    dummy_table = removevars(dummy_table,'unknown');
  end
  new_data = dummy_table;