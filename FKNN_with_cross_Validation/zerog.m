  function y = zerog(x)
% keywords: version compability
% function y = zerog(x)
% the function produces a matrix of zeros with the same
% size as the matrix 'x'.
% NOTE. Only for compatibility between Matlab3.5 and Matlab4.0

y = zeros(length(x(:,1)),length(x(1,:)));
