function [Mean_accuracy, Variance,p1,m1,pp,mm,id,maxsen,Mean_sensitivity,Mean_specificity,Mean_FPR,Mean_FNR]=simclass4(data,v,c, measure, p, m, N,rn,pl)
%Inputs:
% data: put your data file in matrix form, rows indicates samples and
% columns features of the data.
% v:   how many columns of features you are using e.g. [1:4] means you will
% use first four columns as features of your data.
% c: column where you class labels are. They should be in numerical form.
% measure: which similarity measure you want to use measure=1 means
% similarity based on generalized Lukasiewics structure.
% p: parameter in generalized Lukasiewics similarity, can be studied as a range of
% parameter values and then given as a vector i.e. p=[0.1:0.1:5].
% m: generalized mean parameter m=1 means arithmetic mean, m=-1 harmonic
% mean etc. Can be given as vector i.e. m=[0.1:0.1:5].
% N: how many times data is divided randomly into testing set and learning
% set.
% rn: portion of data which is used in learning set. default rn=1/2 (data
% is divided in half; half is used in testing set and half in learning set.
%pl: do you want to plot how the parameter changes in p and m changes the
% mean classification accuracies and variances.
%
%OUTPUTS:
% Mean_accuracy: Mean classification accuracy with best parameter values in
% p and m:
% p1 and m1: best parameter values w.r.t. mean classification accuracy.
% Variance: Variances with best parameter values
% pp,mm,id: Best parameter and ideal vector values from the results with validation set 
%(or first ones of the best results in case of several).
%

sv_name = 'results2'; % mat file where results are stored ('' = "no storing")

[data, lc, cs] = init_data(data,v,c); % data initialization

w_opt = 0; % Do we use weight optimization. Not implemented in this version.

%Initializations
fitness = zeros(1,N);
fitness_id = zeros(1,N);
fitness_dif = zeros(1,N);
Means = zeros(length(p),length(m),length(rn));
Vars = zeros(length(p),length(m),length(rn)); 
Maxsf = zeros(length(p),length(m),length(rn));
Minsf = zeros(length(p),length(m),length(rn));

Means_fit_dif = zeros(length(p),length(m),length(rn));
Vars_fit_dir = zeros(length(p),length(m),length(rn)); 
Ideal_var = zeros(length(p),length(m), length(lc),length(rn));

w = ones(1,size(data,2)-1); % weights for similarity measure. Now just set to ones.
maxsen=0; % MODIFIED to max sen
for n = 1 : length(rn)
    rn_ideal = ones(1,length(lc))*rn(n); 
    for j = 1:length(m) 
        for i = 1:length(p)  
            y = [p(i), m(j),measure]; % p and m values and similarity measure          
            for k = 1 : N 
                ideal_ind = [];
                for l = 1 :length(lc) 
                    temp = randperm(lc(l))-1;                
                    ideal_ind = [ideal_ind, cs(l)+temp(1:floor(lc(l)*rn_ideal(l)))]; % learning set indexes
                    data_ind = setxor([1:size(data,1)], ideal_ind); % testing set indexes
                end
                ideals(:,:,k) = idealvectors(data(ideal_ind,:), y); % idealvectors     
                if w_opt == 0  % No weight optimization
                    [fitness(k), class, Simil,TP(k),FP(k),FN(k),TN(k)] = calcfitness2(data(data_ind,:), ideals(:,:,k), y, w);
                    Sensitivity(k) = TP(k)/(TP(k)+FN(k));
                    Specificity(k) = TN(k)/(TN(k)+FP(k));
                    FPR(k) = FP(k) / (FP(k)+TN(k));
                    FNR(k) = FN(k)/(TP(k)+FN(k));
                    if Sensitivity(k)>maxsen % MODIFIED to max sen
                        pp=p(i); % <---- Why pick this as optimal parameter value? 
                        mm=m(j);
                        id=ideals(:,:,k);
                        maxsen=Sensitivity(k); % MODIFIED to max sen
                    end
                    
                end
                if w_opt == 1      %Weight optimization. Not implemented in current release          
                    Y{1}(:,:) = y; 
                    Y{2}(:,:) = data(ideal_ind,:);
                    Y{3}(:,:) = ideals(:,:,k);
                    [w,bestval,nfeval,iter]=evolut('evol_fitness',VTR,D,XVmin,XVmax,Y,NP,itermax,maxnoprogress,Fstep,CR,strategy,refresh);
                    fitness_id(k) = 1-bestval; 
                    [fitness(k), class, Simil] = calcfitness(data(data_ind,:), ideals(:,:,k), y, w); 
                    fitness_dif(k) = fitness_id(k)-fitness(k); 
                end 
            end
            Means(i,j,n) = mean(fitness);
            Vars(i,j,n) = var(fitness);
            Maxsf(i,j,n) = max(fitness);
            Minsf(i,j,n) = min(fitness);
            Sens(i,j,n) = mean(Sensitivity);
            Spes(i,j,n) = mean(Specificity);
            FPRs(i,j,n) = mean(FPR);
            FNRs(i,j,n) = mean(FNR);
            
            
            fitness=[];
            fprintf('P: %2.0f out of %2.0f \n',i,length(p)); % modified to check calculation process
        end
        fprintf('M: %2.0f out of %2.0f \n',j,length(m)); % modified to check calculation process
    end
    tmp=max(max(Sens)); % modifying this provides other optimal parameter values
    [p1,m1]=find(tmp==Sens); % Use same variable as above to find the index of optimal value
    Mean_accuracy=Means(p1,m1);
    Mean_sensitivity = Sens(p1,m1);
    Mean_specificity = Spes(p1,m1);
    Mean_FPR = FPRs(p1,m1);
    Mean_FNR = FNRs(p1,m1);
    Variance=Vars(p1,m1);
    if pl==1
    [X,Y] = meshgrid(m,p);
    figure
    
    subplot(2,2,1);
    hold on
    surfc(X,Y,Vars(:,:,n))
    plot3(m(m1),p(p1),Vars(p1,m1),'ro','LineWidth',7.0,'MarkerFaceColor','r') % plotting the optimal parameters which come from highest sensitivity AVERAGE
    plot3(mm,pp,Vars(find(p == pp),find(m == mm)),'bo','LineWidth',7.0,'MarkerFaceColor','b') % plotting the optimal parameter out of single division to train and valid set
    title('Variances')
    hold off
    
    subplot(2,2,2)
    hold on
    surfc(X,Y,Means(:,:,n))
    plot3(m(m1),p(p1),Means(p1,m1),'ro','LineWidth',7.0,'MarkerFaceColor','r')
    plot3(mm,pp,Means(find(p == pp),find(m == mm)),'bo','LineWidth',7.0,'MarkerFaceColor','b') % plotting the optimal parameter out of single division to train and valid set
    title('Mean classification accuracies')
    xlabel('m-values')
    ylabel('p-values')
    zlabel('Classification accuracy')
    hold off
    
    subplot(2,2,3)
    hold on
    surf(X,Y,Sens(:,:,n))
    plot3(m(m1),p(p1),Sens(p1,m1),'ro','LineWidth',7.0,'MarkerFaceColor','r')
    plot3(mm,pp,Sens(find(p == pp),find(m == mm)),'bo','LineWidth',7.0,'MarkerFaceColor','b') % plotting the optimal parameter out of single division to train and valid set
    title('Sensitivities')
    xlabel('m-values')
    ylabel('p-values')
    hold off
    
    subplot(2,2,4)
    hold on
    surf(X,Y,Spes(:,:,n))
    plot3(m(m1),p(p1),Spes(p1,m1),'ro','LineWidth',7.0,'MarkerFaceColor','r')
    plot3(mm,pp,Spes(find(p == pp),find(m == mm)),'bo','LineWidth',7.0,'MarkerFaceColor','b') % plotting the optimal parameter out of single division to train and valid set
    title('Specificities')
    xlabel('m-values')
    ylabel('p-values')
    hold off
    end
    clear Y
end
if length(sv_name) ~= 0
    save(sv_name)
end