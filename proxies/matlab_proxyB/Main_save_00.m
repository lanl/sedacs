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
Efield = 1*[-0.40,0.3,-0.31]';
Te = 100 % Some electronic temperature
A = importdata('COORD.dat'); 
TYPE = A.textdata(:);
RX = A.data(:,1); RY = A.data(:,2); RZ = A.data(:,3);
LBox(1) = 14; LBox(2) = 14; LBox(3) = 14 % PBC

% Get Hamiltonian, Overlap, atomic DM = D0 (vector only),  etc
[H0,S,D0,H_INDEX_START,H_INDEX_END,Element_Type,Mnuc,Znuc,Hubbard_U] = H0_and_S(TYPE,RX,RY,RZ,LBox,Nats);
Z = S^(-1/2);
Z0 = Z; S0 = S;
HDIM = max(size(H0));
Rcut = 10.42; Coulomb_acc = 10e-7; TIMERATIO = 10;

% Get Coulomb Matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
[nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
C = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nats,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);

% SCF ground state optimization for H and D and q and occupation factors f, D*S*D = D, Tr[DS] = Nocc, f in [0,1]
[H,Hcoul,Hdipole,D,q,f] = SCF(H0,S,Efield,C,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Nocc,Hubbard_U,Znuc,Nats,Te);
[Etot,Eband0,Ecoul,Edipole,S_ent] = Energy(H0,Hubbard_U,Efield,D0,C,D,q,RX,RY,RZ,f,Te); % Energy calculation - 2*Te*S_ent

%%%%%%%% Num Test Forces

%%%%%%%%%%%%%%%%%%%%%%%%

dx = 0.0001;
[dSx,dSy,dSz] = GetdS(Nats,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox);
[dHx,dHy,dHz] = GetdH(Nats,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox);
[dCx,dCy,dCz] = GetdC(Nats,dx,Coulomb_acc,Rcut,TIMERATIO,HDIM,Hubbard_U,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox);

[Ftot,Fcoul,Fband0,Fdipole,FPulay,FScoul,FSdipole] = Forces(H,H0,S,C,D,D0,dHx,dHy,dHz,dSx,dSy,dSz,dCx,dCy,dCz,Efield,Hubbard_U,q,RX,RY,RZ,Nats,H_INDEX_START,H_INDEX_END);

% Zero Field
% Ftot(1,3) = -2.640524966084072
% q(1:3)' = 0.299074107847914  -0.075231905026516  -0.074189178585703

Force_11 =  Ftot(1,1) 

RX0 = RX;
%%%%%%%% FINITE DIFF FORCE
RX = RX0; RX(1) = RX0(1) + 0.0001;
% Get Hamiltonian, Overlap, atomic DM = D0 (vector only),  etc
[H0,S,D0,H_INDEX_START,H_INDEX_END,Element_Type,Mnuc,Znuc,Hubbard_U] = H0_and_S(TYPE,RX,RY,RZ,LBox,Nats);
Z = S^(-1/2);

% Get Coulomb Matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
[nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
C = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nats,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);

% SCF ground state optimization for H and D and q and occupation factors f, D*S*D = D, Tr[DS] = Nocc, f in [0,1]
[H,Hcoul,Hdipole,D,q,f] = SCF(H0,S,Efield,C,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Nocc,Hubbard_U,Znuc,Nats,Te);
[Etot_p,Eband0,Ecoul,Edipole_p,S_ent] = Energy(H0,Hubbard_U,Efield,D0,C,D,q,RX,RY,RZ,f,Te); % Energy calculation - 2*Te*S_ent
Edipole_p

RX = RX0; RX(1) = RX0(1) - 0.0001;
% Get Hamiltonian, Overlap, atomic DM = D0 (vector only),  etc
[H0,S,D0,H_INDEX_START,H_INDEX_END,Element_Type,Mnuc,Znuc,Hubbard_U] = H0_and_S(TYPE,RX,RY,RZ,LBox,Nats);
Z = S^(-1/2);

% Get Coulomb Matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
[nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
C = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nats,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);

% SCF ground state optimization for H and D and q and occupation factors f, D*S*D = D, Tr[DS] = Nocc, f in [0,1]
[H,Hcoul,Hdipole,D,q,f] = SCF(H0,S,Efield,C,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Nocc,Hubbard_U,Znuc,Nats,Te);
[Etot_m,Eband0,Ecoul,Edipole_m,S_ent] = Energy(H0,Hubbard_U,Efield,D0,C,D,q,RX,RY,RZ,f,Te); % Energy calculation - 2*Te*S_ent
Edipole_m

dETOT = (Etot_p - Etot_m)/.0002
%dFDipole = (Edipole_p - Edipole_m)/.0002

