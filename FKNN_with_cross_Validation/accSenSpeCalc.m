function [acc,sen,spe] = accSenSpeCalc(predicted,actual)
  tp = 0; fp = 0; tn = 0; fn = 0;
  for i = 1:length(actual)
    if predicted(i) == 2 && actual(i) == 2
      tp = tp +1;
    elseif predicted(i) == 2 && actual(i) == 1
      fp = fp +1;
    elseif predicted(i) == 1 && actual(i) == 1
      tn = tn +1;
    elseif predicted(i) == 1 && actual(i) == 2
      fn = fn +1;
    else
        frpintf('Something went terribly wrong!');
    end
  end
  acc = (tp+tn)/(tp+tn+fp+fn);
  sen = tp/(tp+fn); % found subscriber amount / (found subscriber amount + not found subscriber amount)
  spe = tn/(tn+fp);