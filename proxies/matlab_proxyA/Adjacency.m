function [X] = Adjacency(X)

[N,M] = size(X);
for i = 1:N
for j = 1:M
  if abs(X(i,j)) > 0
     X(i,j) = 1;
  end
end
end
