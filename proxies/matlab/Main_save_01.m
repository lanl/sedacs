%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   SCF-TB - PROXY APPLICATION                      %
%                   A.M.N. Niklasson, T1, LANL                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Total Energy:                                                     %
% E = 2Tr[H0(D-D0)] + (1/2)sum_{ij} q_i C_{ij} q_j - Efield*dipole  %
% dipole = sum_i R_{i} q_i                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

% Initial data, load atoms and coordinates, etc
Nats = 50; % Number of atoms 
Nocc = 40;  % Nr of electrons / 2
Efield = 0.0*[0.90,0.0,-.00]';
Te = 100 % Some electronic temperature
A = importdata('COORD.dat'); 
TYPE = A.textdata(:);
RX = A.data(:,1); RY = A.data(:,2); RZ = A.data(:,3);
LBox(1) = 14; LBox(2) = 14; LBox(3) = 14 % PBC

% Get Hamiltonian, Overlap, atomic DM = D0 (vector only),  etc
[H0,S,D0,H_INDEX_START,H_INDEX_END,Element_Type,Mnuc,Znuc,Hubbard_U] = H0_and_S(TYPE,RX,RY,RZ,LBox,Nats);
HDIM = max(size(H0));
Z = S^(-1/2);
Z0 = Z; S0 = S;
Rcut = 10.42; Coulomb_acc = 10e-7; TIMERATIO = 10;

% Get Coulomb Matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
[nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
C = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nats,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);

% SCF ground state optimization for H and D and q and occupation factors f, D*S*D = D, Tr[DS] = Nocc, f in [0,1]
[H,Hcoul,Hdipole,D,q,f] = SCF(H0,S,Efield,C,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Nocc,Hubbard_U,Znuc,Nats,Te);
[Etot,Eband0,Ecoul,Edipole,S_ent] = Energy(H0,Hubbard_U,Efield,D0,C,D,q,RX,RY,RZ,f,Te); % Energy calculation - 2*Te*S_ent

dx = 0.0001;
[dSx,dSy,dSz] = GetdS(Nats,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox);
[dHx,dHy,dHz] = GetdH(Nats,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox);
[dCx,dCy,dCz] = GetdC(Nats,dx,Coulomb_acc,Rcut,TIMERATIO,HDIM,Hubbard_U,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox);

[Ftot,Fcoul,Fband0,Fdipole,FPulay,FScoul,FSdipole] = Forces(H,H0,S,C,D,D0,dHx,dHy,dHz,dSx,dSy,dSz,dCx,dCy,dCz,Efield,Hubbard_U,q,RX,RY,RZ,Nats,H_INDEX_START,H_INDEX_END);

Force_11 =  Ftot(1,1) 

RX0 = RX;
%%%%%%%% FINITE DIFF FORCE

DD = D;
q0 = Get_q(DD,S,H_INDEX_START,H_INDEX_END,Znuc,Nats);
CC = C; RX0 = RX; RY0 = RY; RZ0 = RZ; HH0 = H0; HH = H;

RX = RX0; RX(1) = RX0(1) + 0.0001;
[nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);

[H0,S,D0,H_INDEX_START,H_INDEX_END,Element_Type,Mnuc,Znuc,Hubbard_U] = H0_and_S(TYPE,RX,RY,RZ,LBox,Nats);
C = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nats,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);
[H,Hcoul,Hdipole,D,q,f] = SCF(H0,S,Efield,C,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Nocc,Hubbard_U,Znuc,Nats,Te);
[Etot_p,Eband0_p,Ecoul_p,Edipole_p,S_ent] = Energy(H0,Hubbard_U,Efield,D0,C,D,q,RX,RY,RZ,f,Te); % Energy calculation - 2*Te*S_ent
q_p = q; H_p = H; D_p = D;

RX = RX0; RX(1) = RX0(1) - 0.0001;
[nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);

[H0,S,D0,H_INDEX_START,H_INDEX_END,Element_Type,Mnuc,Znuc,Hubbard_U] = H0_and_S(TYPE,RX,RY,RZ,LBox,Nats);
C = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nats,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);
[H,Hcoul,Hdipole,D,q,f] = SCF(H0,S,Efield,C,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Nocc,Hubbard_U,Znuc,Nats,Te);
[Etot_m,Eband0_m,Ecoul_m,Edipole_m,S_ent] = Energy(H0,Hubbard_U,Efield,D0,C,D,q,RX,RY,RZ,f,Te); % Energy calculation - 2*Te*S_ent
q_m = q; H_m = H; D_m = D;

d_ETOT = (Etot_p - Etot_m)/.0002
Ftot(1,1)

pause
d_EBand0 = (Eband0_p-Eband0_m)/.0002

d_EBand = (2*trace(H_p*(D_p-diag(D0))) - 2*trace(H_m*(D_m-diag(D0))))/.0002

d_EDipole = (Edipole_p - Edipole_m)/.0002
d_Ecoul = (Ecoul_p-Ecoul_m)/.0002

band0 = Fband0(1,1)
F_total = Ftot(1,1)
F_diople = Fdipole(1,1) + FSdipole(1,1)
FS_dipole = FSdipole(1,1)
F_coul = Fcoul(1,1) + FScoul(1,1)
FS_coul = FScoul(1,1)
F_pul = FPulay(1,1)
HEJ = 1
pause

RX = RX0; RX(1) = RX0(1) + 0.0001;
Edipole_p = 0;
for i = 1:Nats
  Edipole_p = Edipole_p - q0(i)*(RX(i)*Efield(1)+RY(i)*Efield(2)+RZ(i)*Efield(3));  % External-field-Dipole interaction energy
end
RX = RX0; RX(1) = RX0(1) - 0.0001;
Edipole_m = 0;
for i = 1:Nats
  Edipole_m = Edipole_m - q0(i)*(RX(i)*Efield(1)+RY(i)*Efield(2)+RZ(i)*Efield(3));  % External-field-Dipole interaction energy
end
d_FDipole = (Edipole_p - Edipole_m)/.0002

dq_dr = (q_p - q_m)/.0002;
d_Edipole_m = 0;
for i = 1:Nats
  d_Edipole_m = d_Edipole_m - dq_dr(i)*(RX(i)*Efield(1)+RY(i)*Efield(2)+RZ(i)*Efield(3));  % External-field-Dipole interaction energy
end
d_EDipole =  d_Edipole_m

F11 = Fband0(1,1) + Fcoul(1,1) + FPulay(1,1) + FScoul(1,1)
