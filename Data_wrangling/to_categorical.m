function new_data = to_categorical(variable, names,exclude_unknown,unknown_name)

  variable = categorical(variable);
  X = dummyvar(variable);
  
  if length(names) == 0
    names = categories(variable);
  end
  
  dummy_table = array2table(X);
  dummy_table.Properties.VariableNames = names;
  
  
  if any(strcmp(names,'unknown')) == 1 && exclude_unknown == 1
    dummy_table = removevars(dummy_table,{'unknown'});
  elseif exclude_unknown == 0
    dummy_table.Properties.VariableNames{'unknown'} = unknown_name;
  end
  new_data = dummy_table;