function [scores,loads]=frpca(x,LV,method)
% Robust PCA in terms of fuzzy PCA 
% frpca1, frpca2 and frpca3 are based on 
%      T.-N. Yang, S.-D. Wang, Robust Algorithms for Principal Component Analysis, Patt. Recogn. Lett., 20 (1999) 927-933.

x=x';
[m,n]=size(x);

if nargin<3
   method='frpca3'
elseif nargin<2
   LV=min(m,n)
end

scores=[]; loads=[];

ALFA0=0.5; % learning coefficient (0,1]
ETA=.1;  % soft threshold, a small positive number
EXPO=2;  % fuzziness variable
%EXPO=2;  % fuzziness variable

if strcmpi(method,'frpca1')==1
   for lv=1:LV   
      alfa0=ALFA0; % learning coefficient (0,1]
      eta=ETA;  % soft threshold, a small positive number
      expo=EXPO;  % fuzziness variable
      t=1;     % iteration count
      T=100;   % iteration bound
      %beta = membership, penalty
      [tt,s,pp] = svds(x',1); w=pp; % initialize loadings
      %w=rand(m,1); % initialize loadings
      wold=zeros(size(w));
      while t<T  % do not add '=' sign
         alfat=alfa0*(1-t/T);
         sigma=0; i=1;
         wold=w;
         while i<n & sum((wold-w).^2)/m>1e-6
            y=w'*x(:,i); u=y*w; v=w'*u;
            e1x=(x(:,i)-w'*x(:,i)*w)'*(x(:,i)-w'*x(:,i)*w);
            beta=(1/(1+(e1x)/eta).^(1/(expo-1))).^expo;
            w=w+alfat*beta*(y*(x(:,i)-u)+(y-v)*x(:,i));
            w=w/norm(w);
            sigma=sigma+e1x;
            i=i+1;
         end
%         t,w
         eta=sigma/n;
         t=t+1;
         if sum((w-wold).^2)/m<1e-6
            break
         end
      end
      scores=[scores x'*w];
      loads=[loads w];
      x=(x'-x'*w*w')';
   end
elseif strcmpi(method,'frpca2')==1
   for lv=1:LV   
      alfa0=ALFA0; % learning coefficient (0,1]
      eta=ETA;  % soft threshold, a small positive number
      expo=EXPO;  % fuzziness variable
      t=1;     % iteration count
      T=100;   % iteration bound
      %beta = membership, penalty
      [tt,s,pp] = svds(x',1); w=pp; % initialize loadings
      %w=rand(m,1); % initialize loadings
      wold=zeros(size(w));
      while t<T  % do not add '=' sign
         alfat=alfa0*(1-t/T);
         sigma=0; i=1;
         wold=w;
         while i<n & sum((wold-w).^2)/m>1e-6
            y=w'*x(:,i); u=y*w; v=w'*u;
            e1x=(x(:,i)-w'*x(:,i)*w)'*(x(:,i)-w'*x(:,i)*w);
            beta=(1/(1+(e1x)/eta).^(1/(expo-1))).^expo;
            w=w+alfat*beta*(x(:,i)*y-((w/(w'*w))*(y.^2)));
            w=w/norm(w);
            aa=x(:,i)'*x(:,i);
            bb=(w'*x(:,i))*(x(:,i)'*w);
            cc=w'*w;
            e2x=aa-bb/cc;
            sigma=sigma+e2x;
            i=i+1;
         end
%         t,w
         eta=sigma/n;
         t=t+1;
         if sum((w-wold).^2)/m<1e-6
            break
         end
      end
      scores=[scores x'*w];
      loads=[loads w];
      x=(x'-x'*w*w')';
   end
elseif strcmpi(method,'frpca3')==1
   for lv=1:LV   
      alfa0=ALFA0; % learning coefficient (0,1]
      eta=ETA;  % soft threshold, a small positive number
      expo=EXPO;  % fuzziness variable
      t=1;     % iteration count
      T=100;   % iteration bound
      %beta = membership, penalty
      [tt,s,pp] = svds(x',1); w=pp; % initialize loadings
      %w=rand(m,1); % initialize loadings
      wold=5*ones(size(w));
      crit=sum((w-wold).^2);
      while t<T % do not add '=' sign
         alfat=alfa0*(1-t/T);
         sigma=0; i=1;
         wold=w;
         while i<=n
            y=w'*x(:,i); 
            e1x=(x(:,i)-w'*x(:,i)*w)'*(x(:,i)-w'*x(:,i)*w);
            beta=(1/(1+(e1x)/eta)^(1/(expo-1)))^expo;
            w=w+alfat*beta*(x(:,i)*y-w*(y.^2));
            w=w/norm(w);
            sigma=sigma+e1x;
            i=i+1;
         end
         eta=sigma/n;
         t=t+1;
%         figure(1)
         crit=[crit,sum((w-wold).^2)/m];
%         semilogy(crit)
%         drawnow
         if sum((w-wold).^2)/m<1e-8
         %   pause(5)
            break
         end
      end
      scores=[scores x'*w];
      loads=[loads w];
      x=x';
      x=x-x*w*w';
      x=x';
   end
end   

function Y=meanw(X,W)
% MEANW : Weighted Mean
%
% Y=meanw(X,W)
% 
% Y is the mean of X weighted by W.
%
if(size(X)~=size(W)), error('Must be 1:1 corrospondence between weights and samples'); end;
Y = sum(W.*X)./sum(W);

function Y=stdw(X,W)
% STDW : Weighted Standard Deviation
%
% Y=stdw(X,W)
% 
% Y is the standard deviation of X weighted by W.
%
if(size(X)~=size(W)), error('Must be 1:1 corrospondence between weights and samples'); end;
Y = sqrt( (sum(W.*X.*X)./sum(W)) - (sum(W.*X)./sum(W)).^2);


