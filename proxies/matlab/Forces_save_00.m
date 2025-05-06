function [Ftot,Fcoul,Fband0,Fdipole,FPulay,FScoul,FSdipole] = Forces(H,H0,S,C,D,D0,dHx,dHy,dHz,dSx,dSy,dSz,dCx,dCy,dCz,Efield,U,q,Rx,Ry,Rz,Nats,H_INDEX_START,H_INDEX_END)

HDIM = max(size(H0));
Fcoul = zeros(3,Nats);
for i = 1:Nats
  Fcoul(1,i) = -q(i)*q(:)'*dCx(:,i);
  Fcoul(2,i) = -q(i)*q(:)'*dCy(:,i);
  Fcoul(3,i) = -q(i)*q(:)'*dCz(:,i);
end

Fband0 = zeros(3,Nats);
for i = 1:Nats % Slater-Koster Force SKForce from Tr[D*dH0/dR]
  I_A = H_INDEX_START(i);
  I_B = H_INDEX_END(i);
  Xtmp = dHx(I_A:I_B,:)*D(:,I_A:I_B);
  Ytmp = dHy(I_A:I_B,:)*D(:,I_A:I_B);
  Ztmp = dHz(I_A:I_B,:)*D(:,I_A:I_B);
  Fband0(1,i) = -2*2*trace(Xtmp);
  Fband0(2,i) = -2*2*trace(Ytmp);
  Fband0(3,i) = -2*2*trace(Ztmp);
end

FSdipole = zeros(3,Nats);
for k = 1:Nats % Slater-Koster Force SKForce from Tr[D*dH0/dR]
  k_a = H_INDEX_START(k);
  k_b = H_INDEX_END(k);
  d_Sx = zeros(HDIM);
  d_Sx(:,k_a:k_b) = dSx(:,k_a:k_b); d_Sx(k_a:k_b,:) = dSx(:,k_a:k_b)';
  d_Sy = zeros(HDIM);
  d_Sy(:,k_a:k_b) = dSy(:,k_a:k_b); d_Sy(k_a:k_b,:) = dSy(:,k_a:k_b)';
  d_Sz = zeros(HDIM);
  d_Sz(:,k_a:k_b) = dSz(:,k_a:k_b); d_Sz(k_a:k_b,:) = dSz(:,k_a:k_b)';

  dqi_dRkX = zeros(Nats,1); dqi_dRkY = zeros(Nats,1); dqi_dRkZ = zeros(Nats,1);
  for i = 1:Nats
    if i == k
      dqi_dRkX(i) = 2*trace(D(k_a:k_b,:)*dSx(:,k_a:k_b));
      dqi_dRkY(i) = 2*trace(D(k_a:k_b,:)*dSy(:,k_a:k_b));
      dqi_dRkZ(i) = 2*trace(D(k_a:k_b,:)*dSz(:,k_a:k_b));
    else
      i_a = H_INDEX_START(i); i_b = H_INDEX_END(i);
      dqi_dRkX(i) = 2*trace(dSx(i_a:i_b,k_a:k_b)*D(k_a:k_b,i_a:i_b));
      dqi_dRkY(i) = 2*trace(dSy(i_a:i_b,k_a:k_b)*D(k_a:k_b,i_a:i_b));
      dqi_dRkZ(i) = 2*trace(dSz(i_a:i_b,k_a:k_b)*D(k_a:k_b,i_a:i_b));
    end
    FSdipole(1,k) = FSdipole(1,k) - dqi_dRkX(i)*(Rx(i)*Efield(1)+Ry(i)*Efield(2)+Rz(i)*Efield(3));
    FSdipole(2,k) = FSdipole(2,k) - dqi_dRkY(i)*(Rx(i)*Efield(1)+Ry(i)*Efield(2)+Rz(i)*Efield(3));
    FSdipole(3,k) = FSdipole(3,k) - dqi_dRkZ(i)*(Rx(i)*Efield(1)+Ry(i)*Efield(2)+Rz(i)*Efield(3));
  end
  if k == 1
    check = dqi_dRkX(1:10)'
  end
end

Fdipole = zeros(3,Nats);
for i = 1:Nats
  Fdipole(1,i) = q(i)*Efield(1);  % Forces from External field-dipole interaction
  Fdipole(2,i) = q(i)*Efield(2);  % Forces from External field-dipole interaction
  Fdipole(3,i) = q(i)*Efield(3);  % Forces from External field-dipole interaction
end

Z = S^(-1/2);
SIHD = 2*2*Z*Z'*H*D;  % Pulay Force FPUL from 2Tr[ZZ'HD*dS/dR]
FPulay = zeros(3,Nats);
for i = 1:Nats
  I_A = H_INDEX_START(i);
  I_B = H_INDEX_END(i);
  Xtmp = dSx(I_A:I_B,:)*SIHD(:,I_A:I_B);
  Ytmp = dSy(I_A:I_B,:)*SIHD(:,I_A:I_B);
  Ztmp = dSz(I_A:I_B,:)*SIHD(:,I_A:I_B);
  FPulay(1,i) = trace(Xtmp);
  FPulay(2,i) = trace(Ytmp);
  FPulay(3,i) = trace(Ztmp);
end

CoulPot = C*q;  % Factor of 2 or 1/2 or +/-
FScoul = zeros(3,Nats); % Coulomb force FSCOUL from nonorthogonality
dDSX = zeros(HDIM,1);
dDSY = zeros(HDIM,1);
dDSZ = zeros(HDIM,1);
for Ia = 1:Nats % Derivatives Ra
  Ia_A = H_INDEX_START(Ia);
  Ia_B = H_INDEX_END(Ia);
  for iq = 1:HDIM
    dDSX(iq) = D(iq,Ia_A:Ia_B)*dSx(Ia_A:Ia_B,iq);
    dDSY(iq) = D(iq,Ia_A:Ia_B)*dSy(Ia_A:Ia_B,iq);
    dDSZ(iq) = D(iq,Ia_A:Ia_B)*dSz(Ia_A:Ia_B,iq);
  end
  for iq = Ia_A:Ia_B
    dDSX(iq) = dDSX(iq) + D(iq,:)*dSx(iq,:)';
    dDSY(iq) = dDSY(iq) + D(iq,:)*dSy(iq,:)';
    dDSZ(iq) = dDSZ(iq) + D(iq,:)*dSz(iq,:)';
  end
  for j = 1:Nats % Get the Mulliken charges for all atoms
    j_a = H_INDEX_START(j);
    j_b = H_INDEX_END(j);
    dQLxdR = sum(dDSX(j_a:j_b)); % Derivative with respect to Ia of charge on atom j
    dQLydR = sum(dDSY(j_a:j_b));
    dQLzdR = sum(dDSZ(j_a:j_b));
    FScoul(1,Ia) = FScoul(1,Ia) - dQLxdR*(U(j)*q(j) + CoulPot(j));
    FScoul(2,Ia) = FScoul(2,Ia) - dQLydR*(U(j)*q(j) + CoulPot(j));
    FScoul(3,Ia) = FScoul(3,Ia) - dQLzdR*(U(j)*q(j) + CoulPot(j));
  end

end
FScoul = 2*FScoul;

Ftot  = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole;  % Collected total force
