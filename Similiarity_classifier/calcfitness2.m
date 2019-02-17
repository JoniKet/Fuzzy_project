function [fitness, class, Simil,TP,FP,FN,TN] = calcfitness2(data, ideals, y, w)
[nc, v_dim] = size(ideals);  
d_dim = size(data,1);  

if nargin==3    
   w=ones(1,v_dim);
end
[class, Simil] = classifier(data(:,1:v_dim), ideals, y, w); 
fitness = length(find(class-data(:, v_dim +1) == 0))/d_dim;
[TP,FP,FN,TN]=performancemeasures(class,data(:,v_dim+1));