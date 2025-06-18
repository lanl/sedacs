function [dCx,dCy,dCz] = GetdC(Nr_atoms,dh,Coulomb_acc,Rcut,TIMERATIO,HDIM,Hubbard_U,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox)

  dq_J = zeros(1,Nr_atoms); Nats = Nr_atoms;
  dCx = zeros(Nr_atoms); dCy = zeros(Nr_atoms); dCz = zeros(Nr_atoms);
  Coulomb_Pot_k = 0;
  RX0 = RX; RY0 = RY; RZ0 = RZ;

  for J = 1:Nr_atoms

    dq_J = 0*dq_J;
    dq_J(J) = 1;

    RX(J) = RX0(J) + dh;
    [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
    for I = 1:Nr_atoms
      [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,RX,RY,RZ,LBox,dq_J,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
    end
    [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,dq_J,Nr_atoms,Coulomb_acc,TIMERATIO);
    Coulomb_Pot_p = Coulomb_Pot_Real + Coulomb_Pot_k;
    RX = RX0; 

    RX(J) = RX0(J) - dh;
    [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
    for I = 1:Nr_atoms
      [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,RX,RY,RZ,LBox,dq_J,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
    end
    [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,dq_J,Nr_atoms,Coulomb_acc,TIMERATIO);
    Coulomb_Pot_m = Coulomb_Pot_Real + Coulomb_Pot_k;
    RX = RX0;

    dCx(:,J) = (Coulomb_Pot_p - Coulomb_Pot_m)/(2*dh);

%%%%
%RX0 = RX;
%RX(J) = RX0(J) + dh;
%Cp = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nr_atoms,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);
%RX = RX0;
%
%RX(J) = RX0(J) - dh;
%Cm = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nr_atoms,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);
%RX = RX0;
%dC_x = (Cp-Cm)/(2*dh);
%%%%

    RY = RY0; RY(J) = RY0(J) + dh;
    [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
    for I = 1:Nr_atoms
      [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,RX,RY,RZ,LBox,dq_J,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
    end
    [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,dq_J,Nr_atoms,Coulomb_acc,TIMERATIO);
    Coulomb_Pot_p = Coulomb_Pot_Real + Coulomb_Pot_k;
    RY = RY0; 

    RY(J) = RY0(J) - dh;
    [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
    for I = 1:Nr_atoms
      [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,RX,RY,RZ,LBox,dq_J,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
    end
    [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,dq_J,Nr_atoms,Coulomb_acc,TIMERATIO);
    Coulomb_Pot_m = Coulomb_Pot_Real + Coulomb_Pot_k;
    RY = RY0;

    dCy(:,J) = (Coulomb_Pot_p - Coulomb_Pot_m)/(2*dh);

    RZ(J) = RZ0(J) + dh;
    [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
    for I = 1:Nr_atoms
      [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,RX,RY,RZ,LBox,dq_J,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
    end
    [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,dq_J,Nr_atoms,Coulomb_acc,TIMERATIO);
    Coulomb_Pot_p = Coulomb_Pot_Real + Coulomb_Pot_k;
    RZ = RZ0; 

    RZ(J) = RZ0(J) - dh;
    [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
    for I = 1:Nr_atoms
      [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,RX,RY,RZ,LBox,dq_J,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
    end
    [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,dq_J,Nr_atoms,Coulomb_acc,TIMERATIO);
    Coulomb_Pot_m = Coulomb_Pot_Real + Coulomb_Pot_k;
    RZ = RZ0;

    dCz(:,J) = (Coulomb_Pot_p - Coulomb_Pot_m)/(2*dh);

    dq_J(J) = 0;
  end
