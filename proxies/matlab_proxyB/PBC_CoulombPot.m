function [Vcoul,dVcoul] = PBC_CoulombPot(q,C,U)

 N = max(size(q));
 Vcoul = zeros(N,1);
 dVcoul = zeros(N,N);
 for i = 1:N
   for j = 1:N
     if i~=j
       Vcoul(i) = Vcoul(i) + q(j)/norm(R(i)-R(j));
     else
       Vcoul(i) = Vcoul(i) + q(i)*U(i);
     end
   end
 end

 for k = 1:N
 for i = 1:N
   for j = 1:N
     if j~=i
       if i==k
          dVcoul(i,k) = dVcoul(i,k) - q(j)*(R(i)-R(j))/(norm(R(i)-R(j))^3);
       end
       if j==k
          dVcoul(i,k) = dVcoul(i,k) + q(j)*(R(i)-R(j))/(norm(R(i)-R(j))^3);
       end
     end
   end
 end
 end
