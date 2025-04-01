function [dH0x,dH0y,dH0z] = GetdH(Nr_atoms,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox,nrnnlist,nnRx,nnRy,nnRz,nnType)

dH0x = zeros(HDIM,HDIM); dH0y = zeros(HDIM,HDIM); dH0z = zeros(HDIM,HDIM);
for I = 1:Nr_atoms
  Type_pair(1) = Element_Type(I);
  Rax_p = [RX(I)+dx,RY(I),RZ(I)]; Rax_m = [RX(I)-dx,RY(I),RZ(I)];
  Ray_p = [RX(I),RY(I)+dx,RZ(I)]; Ray_m = [RX(I),RY(I)-dx,RZ(I)];
  Raz_p = [RX(I),RY(I),RZ(I)+dx]; Raz_m = [RX(I),RY(I),RZ(I)-dx];
  IDim = H_INDEX_END(I)-H_INDEX_START(I)+1;
  for J = 1:nrnnlist(I)
    IJ = nnType(I,J);
    if IJ ~= I
      Type_pair(2) = Element_Type(IJ);
      Rb = [nnRx(I,J),nnRy(I,J),nnRz(I,J)];
      JDim = H_INDEX_END(IJ)-H_INDEX_START(IJ)+1;

      [fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,Es,Ep,U] = LoadBondIntegralParameters_H(Type_pair); % Used in BondIntegral(dR,fxx_xx)
      diagonal(1:2) = [Es,Ep];
      dh0x_p = Slater_Koster_Pair(Rax_p,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      dh0x_m = Slater_Koster_Pair(Rax_m,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      dh0y_p = Slater_Koster_Pair(Ray_p,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      dh0y_m = Slater_Koster_Pair(Ray_m,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      dh0z_p = Slater_Koster_Pair(Raz_p,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      dh0z_m = Slater_Koster_Pair(Raz_m,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);

      for II = 1:IDim
        II_H = H_INDEX_START(I) + II - 1;
        for JJ = 1:JDim
          JJ_H = H_INDEX_START(IJ) + JJ - 1;
          dH0x(II_H,JJ_H) = dH0x(II_H,JJ_H) + (dh0x_p(II,JJ)-dh0x_m(II,JJ))/(2*dx);
          dH0y(II_H,JJ_H) = dH0y(II_H,JJ_H) + (dh0y_p(II,JJ)-dh0y_m(II,JJ))/(2*dx);
          dH0z(II_H,JJ_H) = dH0z(II_H,JJ_H) + (dh0z_p(II,JJ)-dh0z_m(II,JJ))/(2*dx);
        end
      end

    end
  end
end
