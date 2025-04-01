function [H0] = PBC_Hamiltonian(R,L,N,Rnd)

av = sum(Rnd)/N;
H0 = zeros(N); 
for i = 1:N
  H0(i,i) = 1*(Rnd(i)-av); 
  for j = i+1:N
    Dist2 = min([(R(i)-R(j))^2,(R(i)-R(j)+L)^2,(R(i)-R(j)-L)^2]);
    %H0(i,j) = (Rnd(i)+Rnd(j))*exp(-3*Dist2); H0(j,i) = H0(i,j);
    H0(i,j) = (Rnd(i)+Rnd(j))*exp(-1*Dist2); H0(j,i) = H0(i,j);
  end
end
