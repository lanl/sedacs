function [dSx,dSy,dSz] = GetdS(Nr_atoms,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox,nrnnlist,nnRx,nnRy,nnRz,nnType)

dSx = zeros(HDIM,HDIM); dSy = zeros(HDIM,HDIM); dSz = zeros(HDIM,HDIM);
for I = 1:Nr_atoms
  Type_pair(1) = Element_Type(I);
  Rax_p = [RX(I)+dx,RY(I),RZ(I)]; Rax_m = [RX(I)-dx,RY(I),RZ(I)];
  Ray_p = [RX(I),RY(I)+dx,RZ(I)]; Ray_m = [RX(I),RY(I)-dx,RZ(I)];
  Raz_p = [RX(I),RY(I),RZ(I)+dx]; Raz_m = [RX(I),RY(I),RZ(I)-dx];
  IDim = H_INDEX_END(I)-H_INDEX_START(I)+1;
  for J = 1:nrnnlist(I)
    IJ = nnType(I,J) ;
    if IJ ~= I
      Type_pair(2) = Element_Type(IJ);
      Rb = [nnRx(I,J),nnRy(I,J),nnRz(I,J)];
      JDim = H_INDEX_END(IJ)-H_INDEX_START(IJ)+1;
      diagonal(1:2) = [1,1];
      [fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi] = LoadBondIntegralParameters_S(Type_pair);
      ds0x_p = Slater_Koster_Pair(Rax_p,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      ds0x_m = Slater_Koster_Pair(Rax_m,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      ds0y_p = Slater_Koster_Pair(Ray_p,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      ds0y_m = Slater_Koster_Pair(Ray_m,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      ds0z_p = Slater_Koster_Pair(Raz_p,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      ds0z_m = Slater_Koster_Pair(Raz_m,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal);
      for II = 1:IDim
        II_S = H_INDEX_START(I) + II - 1;
        for JJ = 1:JDim
          JJ_S = H_INDEX_START(IJ) + JJ - 1;
          dSx(II_S,JJ_S) = dSx(II_S,JJ_S) + (ds0x_p(II,JJ)-ds0x_m(II,JJ))/(2*dx);
          dSy(II_S,JJ_S) = dSy(II_S,JJ_S) + (ds0y_p(II,JJ)-ds0y_m(II,JJ))/(2*dx);
          dSz(II_S,JJ_S) = dSz(II_S,JJ_S) + (ds0z_p(II,JJ)-ds0z_m(II,JJ))/(2*dx);
        end
      end

    end
  end
end

