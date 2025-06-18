function [C] = PBC_Coulomb(R,L,N,U)

C = zeros(N);
for i = 1:N
  C(i,i) = U(i);
  for j = i+1:N
    Dist2 = min([(R(i)-R(j))^2,(R(i)-R(j)+L)^2,(R(i)-R(j)-L)^2]);
    %C(i,j) = (U(i)+U(j))*exp(-4*sqrt(Dist2)); C(j,i) = C(i,j);
    C(i,j) = (U(i)+U(j))*exp(-1*sqrt(Dist2)); C(j,i) = C(i,j);
  end
end
