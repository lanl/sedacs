function [H0,S,D0,H_INDEX_START,H_INDEX_END,Element_Type,Mnuc,Znuc,Hubbard_U] = H0_and_S(TYPE,RX,RY,RZ,LBox,Nr_atoms,nrnnlist,nnRx,nnRy,nnRz,nnType)

%TYPE = A.textdata(:);
%RX = A.data(:,1); RY = A.data(:,2); RZ = A.data(:,3);
CNT = 1;
for i = 1:Nr_atoms
  if char(TYPE(i)) == 'H'
    H_INDEX_START(i) = CNT;
    NrOrb(i) = 1;  % a single s orbital
    CNT = CNT+1;
    H_INDEX_END(i) = CNT-1;
    Znuc(i) = 1; % For hydrogen
    Mnuc(i) = 1.0079;
  else
    H_INDEX_START(i) = CNT;
    NrOrb(i) = 4;  % one 1 + three p orbitals
    CNT = CNT+4;
    H_INDEX_END(i) = CNT-1;
    if char(TYPE(i)) == 'O'
      Znuc(i) = 6; % For oxygen
      Mnuc(i) = 15.9994;
    end
    if char(TYPE(i)) == 'C'
      Znuc(i) = 4; % For oxygen
      Mnuc(i) = 12.01;
    end
    if char(TYPE(i)) == 'N'
      Znuc(i) = 5; % For oxygen
      Mnuc(i) = 14.0067;
    end
  end
  Element_Type(i) = char(TYPE(i));
end
HDIM = CNT-1;

H0 = zeros(HDIM,HDIM); % Charge independent H0!
for I = 1:Nr_atoms

  Type_pair(1) = Element_Type(I);
  Ra = [RX(I),RY(I),RZ(I)];
  IDim = H_INDEX_END(I)-H_INDEX_START(I)+1;
  for J = 1:nrnnlist(I)
    Type_pair(2) = Element_Type(nnType(I,J));
    Rb = [nnRx(I,J),nnRy(I,J),nnRz(I,J)];
    JDim = H_INDEX_END(nnType(I,J))-H_INDEX_START(nnType(I,J))+1;
% Hamiltonian block for a-b atom pair
    [fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,Es,Ep,U] = LoadBondIntegralParameters_H(Type_pair); % Used in BondIntegral(dR,fxx_xx);
    if I == nnType(I,J)
      Hubbard_U(I) = U;
    end
    diagonal(1:2) = [Es,Ep];
    h0  = Slater_Koster_Pair(Ra,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
    for II = 1:IDim
      II_H = H_INDEX_START(I) + II - 1;
      for JJ = 1:JDim
        JJ_H = H_INDEX_START(nnType(I,J)) + JJ - 1;
        H0(II_H,JJ_H) = h0(II,JJ) ;
        H0(JJ_H,II_H) = h0(II,JJ);
      end
    end
  end

end
H0 = 0.5*(H0+H0');

S = zeros(HDIM,HDIM);
for I = 1:Nr_atoms
  Type_pair(1) = Element_Type(I);
  Ra = [RX(I),RY(I),RZ(I)];
  IDim = H_INDEX_END(I)-H_INDEX_START(I)+1;

  for J = 1:nrnnlist(I)
    Type_pair(2) = Element_Type(nnType(I,J));
    Rb = [nnRx(I,J),nnRy(I,J),nnRz(I,J)];
    JDim = H_INDEX_END(nnType(I,J))-H_INDEX_START(nnType(I,J))+1;
% Overlap block for a-b atom pair
    [fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi] = LoadBondIntegralParameters_S(Type_pair);
    diagonal(1:2) = [1,1];
    s0  = Slater_Koster_Pair(Ra,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
    for II = 1:IDim
      II_S = H_INDEX_START(I) + II - 1;
      for JJ = 1:JDim
        JJ_S = H_INDEX_START(nnType(I,J)) + JJ - 1;
        S(II_S,JJ_S) = s0(II,JJ);
        S(JJ_S,II_S) = s0(II,JJ);
      end
    end
  end
end
S = 0.5*(S+S');

D0 = AtomicDensityMatrix(Nr_atoms,H_INDEX_START,H_INDEX_END,HDIM,Znuc);
D0 = D0/2;


