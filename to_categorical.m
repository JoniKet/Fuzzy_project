function new_data = to_categorical(variable)
  variable = categorical(variable);
  X = dummyvar(variable);
  names = categories(variable);
  dummy_table = array2table(X);
  dummy_table.Properties.VariableNames = names;
  
  
  if any(strcmp(names,'unknown')) == 1
    dummy_table = removevars(dummy_table,'unknown');
  end
  new_data = dummy_table;