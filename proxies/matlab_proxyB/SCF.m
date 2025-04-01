function [H,Hcoul,Hdipole,D,Dorth,q,f] = SCF(H0,S,Efield,C,Rx,Ry,Rz,H_Index_Start,H_Index_End,nocc,U,Znuc,Nats,Te)

N = max(size(H0));
it = 0; Res = 1;
Z = S^(-1/2);
h = sort(eig(Z'*H0*Z));
mu0 = 0.5*(h(nocc)+h(nocc+1)); 
[D,mu0] = DM_Fermi(Z'*H0*Z,Te,mu0,nocc,16,1e-9,50);
D = Z*D*Z';
DS = 2*diag(D*S);
q = zeros(Nats,1);
for i = 1:Nats
  q(i) = sum(DS(H_Index_Start(i):H_Index_End(i))) - Znuc(i);
end

while Res > 1e-10
 it = it + 1;
 Dipole = diag(-Rx*Efield(1)-Ry*Efield(2)-Rz*Efield(3));
 Hdipole = zeros(N);
 CoulPot = C*q;

 Hcoul = zeros(N);
 for i = 1:Nats
   for j = H_Index_Start(i):H_Index_End(i)
     Hdipole(j,j) = Dipole(i);
     Hcoul(j,j) = U(i)*q(i) + CoulPot(i);
   end
 end

 Hcoul = 0.5*Hcoul*S + 0.5*S*Hcoul;
 Hdipole = 0.5*Hdipole*S + 0.5*S*Hdipole;
 H = H0 + Hcoul + Hdipole;

 [Dorth,mu0] = DM_Fermi(Z'*H*Z,Te,mu0,nocc,16,1e-9,50);

 D = Z*Dorth*Z';
 q_old = q;

 DS = 2*diag(D*S);
 for i = 1:Nats
   q(i) = sum(DS(H_Index_Start(i):H_Index_End(i))) - Znuc(i);
 end

 Res = norm(q-q_old);

 q = q_old + 0.2*(q-q_old);

end
f = eig(0.5*(Dorth+Dorth'));

