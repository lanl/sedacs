%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calcualte total forces and some matrix derivatives %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ftot,dHdR,dCdR] = PBC_Forces(H0,Efield,D0,C,D,q,R,L,U,Rnd);

N = max(size(q));

dHdR = 0*H0; 
R0 = R; dR = 0.0001;
dEband0_dR = zeros(N,1); 
dEcoul_dR = zeros(N,1); 

for i = 1:N
  Rp = R0;
  Rp(i,1) = Rp(i,1) + dR;           % Finite difference forward displacement
  Hp = PBC_Hamiltonian(Rp,L,N,Rnd);
  Cp = PBC_Coulomb(Rp,L,N,U);
  Rm = R0;
  Rm(i,1) = Rm(i,1) - dR;           % Finite difference backward displacement
  Hm = PBC_Hamiltonian(Rm,L,N,Rnd);
  Cm = PBC_Coulomb(Rm,L,N,U);
  dHdR(:,i) = (Hp(:,i) - Hm(:,i))/(2*dR);  %% H0 derivative collected columnwise. In reality dHdR = dHdR + dHdR'
  dCdR(:,i) = (Cp(:,i) - Cm(:,i))/(2*dR);  %% C derivative collected columnwise. In reality dCdR = dCdR + dCdR'
end                                        %% Otherwise use q(:)'*dCdR(:,i), which is calculated in regular Ewald

for i = 1:N  % force contributions from band energy 2Tr[H(D-D0)] and Coulomb/Hartree energy (1/2) sum_{ij} q_i C_{ij}q_j
  dEband0_dR(i) = 2*2*(D(i,:)-D0(i,:))*dHdR(:,i);
  dEcoul_dR(i) = q(i)*q(:)'*dCdR(:,i);
end

Fdipole = zeros(N,1);
for i = 1:N
  Fdipole(i) = -q(i)*Efield;  % Forces from External field-dipole interaction
end

Ftot  = - dEband0_dR - dEcoul_dR - Fdipole;  % Collected total force
