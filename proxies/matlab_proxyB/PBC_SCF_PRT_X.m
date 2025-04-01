%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCF linear response calculation with respect to atomic displacements  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [H,H_1,Hcoul,Hdipole,dVcoul,D,D1,q] = PBC_SCF_PRT_X(H0,dHdR,C,dCdR,Efield,D0,U,q,R,mu0,nocc,k,Te)

N = max(size(H0));
q_new = q; q_old = 0*q;
q1 = 0*q; q1_new = q1; q1_old = q1;
Vdipole = zeros(N,1);
mu1 = 0;

H1 = zeros(N);
H1(:,k) = dHdR(:,k);
H1 = H1 + H1';

it = 0;
while norm(q-q_old)+norm(q1-q1_old) > 1e-9
 it = it + 1;

 V1coul = C*q1_new;           % Linear response in Hartree/Coulomb potential from charge response due to dsiplacement

 dVcoul = dCdR(:,k)*q_new(k); % Linear response in Hartree/Coulomb potential from displacement only
 dVcoul(k) = dVcoul(k)  + q_new'*dCdR(:,k);  % Combined total linear response in Hartree/Coulomb energy from displacement

 Vcoul = C*q_new;             % Coulomb/Hartree potential 
 Hcoul = diag(Vcoul);         % Coulomb/Hartree Hamiltonian
 Hdipole = diag(-R*Efield);   % Hamiltonian from external field-dipole interaction
 dR = 0*R; dR(k) = 1;

 H_1 = H1 + diag(V1coul) + diag(dVcoul) + diag(-dR*Efield);  % Linear response Hamiltonian
 H = H0 + Hcoul + Hdipole;                                   % Total ground-state Kohn-Sham Hamiltonian
% [D,D1] = DensityMatrixPRT(H,H_1,nocc);  % 2*trace(D) = Ne = 2*nocc, D = density matrix, D1 = response density matrix
% Occ1 = trace(D1)
 [D,D1,mu0,mu1] = DM_PRT_Fermi(H,H_1,Te,mu0,mu1,nocc,16,1e-9,20);
% Occ2 = trace(D1)
% IdFel = norm(D*D-D)
% norm(q-q_old)+norm(q1-q1_old)
% pause

 q1_old = q1_new;
 q1 = 2*diag(D1);                        % Linear response in atom-projected charges
 q1_new = 0.1*q1 + (1-0.1)*q1_old;       % Simple linear mixing

 q_old = q_new;
 q = 2*diag(D-D0);                       % Atomic charges
 q_new = 0.1*q + (1-0.1)*q_old;          % Simple linear mixing
end

