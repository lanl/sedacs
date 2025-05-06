function [CC] = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nr_atoms,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);

  dq_J = zeros(1,Nr_atoms);
  CC = zeros(Nr_atoms);
  Coulomb_Pot_k = 0;
  for J = 1:Nr_atoms
    dq_J(J) = 1;
    for I = 1:Nr_atoms
      [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,RX,RY,RZ,LBox,dq_J,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
    end
    [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,dq_J,Nr_atoms,Coulomb_acc,TIMERATIO);
    Coulomb_Pot_dq_J = Coulomb_Pot_Real+Coulomb_Pot_k;
    CC(:,J) = Coulomb_Pot_dq_J;
    dq_J(J) = 0;
  end
%  CC = 0.5*(CC+CC');
