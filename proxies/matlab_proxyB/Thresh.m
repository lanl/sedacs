function [X] = Thresh(X,eps);

N = max(size(X));
for i = 1:N
for j = 1:N
   if abs(X(i,j)) < eps
      X(i,j) = 0;
   end  
end
end
