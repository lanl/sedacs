%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dual Susceptibility approach to calculate Born-Effective charges  %
%   1-dimensional DFTB example with periodic boundary conditions    %
%                   A.M.N. Niklasson, T1, LANL                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Todo: Implement in DFTB/LATTE, which includes                     %
%     0) Extenson from 1D to 3D with x,y, and z                     %
%     1) Extension to fractional occupation Te > 0, Done!           %
%     2) Extension to general non-orthonormal basis sets            %
%     3) If possible, do Periodic Boundary Conditions correctly,    %
%        without results depending on the edges in Hdipole, Correct!%
%     4) Implement response calculation with AI hardware/GPU        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Total Energy:                                                     %
% E = 2Tr[H0(D-D0)] + (1/2)sum_{ij} q_i C_{ij} q_j - Efield*dipole  %
% dipole = sum_i R_{i} q_i, half filled, 1 basis-function/atom      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
N = 20; % N > 2  % Number of atoms and basis functions
Te = 50000 % Some electronic temperature
Efield = 0.0000;     % External field +/- dEfield give the BEQ from force derivative
atoms = [1:N];   % Atomic positions
R0 = atoms;
atoms = [N-1:N,1:N-2];   % Atomic positions shifted cyclically => same BEQ but shifted
SEED = atoms; Rnd = (sin(2.4+3*SEED(:).^3-sin(pi*SEED(:)/N))); % "Randomized" seed for models
Rnd2 = (10 + 1*(sin(12+3.3*SEED(:).^5-sin(pi*SEED(:))))); % "Randomized" seed for models
R = atoms'; L = N; %R = R + 3.3;
U = Rnd2;
D0 = eye(N)/2; nocc = N/2;                                 % Atomic denity matrix 
H0 = PBC_Hamiltonian(R,L,N,Rnd); C = PBC_Coulomb(R,L,N,U); % Hamiltonian H0 and Coulomb matrix C, toy models

n = N; M = 1;  % M is the number of unit cells
Rtmp = R; Rndtmp = Rnd;  Rndtmp2 = Rnd2;
for i = 1:M
  R((i-1)*n+1:i*n) = (i-1)*L + Rtmp(1:n);
  Rnd((i-1)*n+1:i*n) = Rndtmp(1:n);
  Rnd2((i-1)*n+1:i*n) = Rndtmp2(1:n);
end
N = n*M; L = N; nocc = N/2; D0 = eye(N)/2; 
U = Rnd2;
H0 = PBC_Hamiltonian(R,L,N,Rnd); C = PBC_Coulomb(R,L,N,U); % Hamiltonian H0 and Coulomb matrix C, toy models

[D,mu0] = DensityMatrix(H0,nocc); q = 2*(diag(D-D0));          % Density matrix and first initial charge guess
[D,mu0]  = DM_Fermi(H0,Te,mu0,nocc,16,1e-9,100);
[H,Hcoul,Hdipole,D,q] = PBC_SCF(H0,Efield,D0,C,q,R,nocc,Te);  % Self-consistent optimization

[Etot,Eband0,Ecoul,Edipole,S_ent] = PBC_Energy(H0,Efield,D0,C,D,q,R,Te); % Energy calculation - 2*Te*S_ent
[Ftot,dHdR,dCdR] = PBC_Forces(H0,Efield,D0,C,D,q,R,L,U,Rnd);    % Forces and H0 and C derviative matrices (column wise)

A = diag(R0);                 % Position operator
a_dipole = 2*trace(A*(D-D0)) % Dipole
alt_a_dipole = R'*q
pause

%%% Calculate response in dipole with respect to atomic displacement, i.e. where d_dipole/dR_k = d^2_E//(dRk dEfield)
%%%% Use direct perturbation with respect to displacement, calculated sepconsistently as in DF-PRT
for k = 1:N  
  [H,H_1,Hcoul,Hdipole,dVcoul,D,D1,q] = PBC_SCF_PRT_X(H0,dHdR,C,dCdR,Efield,D0,U,q,R,mu0,nocc,k,Te); % N number of SCF_PRT, one for each k!!!
%  ![H,H_1,Hcoul,Hdipole,dVcoul,D,D1,q] = PBC_SCF_PRT_X(H0,dHdR,C,dCdR,Efield,D0,U,q,R0,mu0,nocc,k,Te); % N number of SCF_PRT, one for each k!!!
q1 = diag(D1);
[k,q(1:10)'];
[k,q1(1:10)'];

  dA = 0*A; dA(k,k) = 1;
  dadR(k) = 2*trace(A*D1) + 2*trace(dA*(D-D0));  % d_dipole/dR_k = d^2_E//(dRk dEfield) = 2*R'*q1 + 2*q(k);
  Same = [dadR(k) ,2*R0*q1 + q(k)];
end
a_dipole = R'*q
a_dipole_R0 = R0*q

%% Use dual SCF susceptibility approach to DF-PRT with respect to dipole observable A
%% Susceptibility for dipole observable -> XA
A0 = A;
Q = eye(N)-D;
%[H,H_1,Hcoul,Hdipole,D,XA,q] = PBC_SCF_PRT(H0,A,C,Efield,D0,U,q,R,mu0,nocc,Te); % Only one SCF_PRT using the dual susceptibility approach!!!
[H,H_1,Hcoul,Hdipole,D,XA,q] = PBC_SCF_PRT(H0,A,C,Efield,D0,U,q,R0,mu0,nocc,Te); % Only one SCF_PRT using the dual susceptibility approach!!!
%[H,H_1,Hcoul,Hdipole,D,XA,q] = PBC_SCF_PRT_0(H0,A,C,Efield,D0,U,q,R,nocc);  % At Te = 0

for k = 1:N  %% Calculate dipole response using XA susceptibility 
  dA = 0*A; dA(k,k) = 1;
  dVcoul = zeros(N,1); dR = zeros(N,1); dR(k) = 1;
  dH = zeros(N);
  dH(:,k) = dHdR(:,k);
  dH = dH + dH';
  
  dVcoul = dCdR(:,k)*q(k); dVcoul(k) = dVcoul(k)  + q'*dCdR(:,k);
  dH = dH + diag(dVcoul) + diag(-dR*Efield);

  da_dR(k) = 2*trace(XA*dH) + 2*trace(dA*(D-D0)); % d_dipole/dR_k = d^2_E//(dRk dEfield) from susceptibility calculation
end

a_dipole = 2*trace(A*(D-D0))  % Still the same!

% Check equivalence between the direct DF-PRT approach and the dual susceptibility formulation
Reldiff = norm(dadR-da_dR)/norm(dadR)
%q'
mm = 2
dadR
da_dR
%da_dR(n*(mm-1)+1:mm*n)
%da_dR(n*mm+1:(mm+1)*n)
