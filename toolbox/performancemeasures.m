function [TP,FP,FN,TN]=perfmeasures(ypred,yreal)

n=length(ypred);
TP=0; FP=0; FN=0; TN=0;
for i=1:n
   if (ypred(i)==2 & yreal(i)==2)
       TP=TP+1;
   elseif (ypred(i)==1 & yreal(i)==1)
       TN=TN+1;
   elseif (ypred(i)==2 & yreal(i)==1)
       FP=FP+1;
   elseif (ypred(i)==1 & yreal(i)==2)
       FN=FN+1;
   end       
end