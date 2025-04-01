function [D,D1] = DensityMatrixPRT(H0,H1,nocc)

N = max(size(H0));
[Q,E] = eig(H0);
e = diag(E);
mu = 0.5*(e(nocc)+e(nocc+1));
D = zeros(N);
for i = 1:N
  if e(i) < mu
     D = D + Q(:,i)*Q(:,i)';
  end
end

Q1 = 0*Q;
for i = 1:N
  if e(i) < mu
    for j = 1:N
         H1Tmp = (Q(:,j)'*H1*Q(:,i)/(e(i) -e(j)));
       if j ~= i
         Q1(:,i) = Q1(:,i) + H1Tmp*Q(:,j);
       end
    end
  end
end
D1 = Q1*Q' + Q*Q1';  % Calculated using Rayleigh Schrodinger Pertrubations theory

% OccErr1 = trace(D1)-0
% IdErr1 = norm(D1*D+D*D1-D1)
% ComErr1 = norm(D1*H0-H0*D1 + D*H1-H1*D)
% pause




