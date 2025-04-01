function [D,mu] = DensityMatrix(H,nocc)

N = max(size(H));
[Q,E] = eig(H);
e = sort(diag(E));
mu = 0.5*(e(nocc)+e(nocc+1));
D = zeros(N);
for i = 1:N
  if e(i) < mu
     D = D + Q(:,i)*Q(:,i)';
  end
end

