function [COULOMBV,FCOUL] = Ewald_Real_Space(I,RX,RY,RZ,LBox,DELTAQ,U,Element_Type,Nr_atoms,COULACC,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType)

ccnt = 0;
COULVOL = LBox(1)*LBox(2)*LBox(3);
SQRTX = sqrt(-log(COULACC));
CALPHA = sqrt(pi)*((TIMERATIO*Nr_atoms/(COULVOL^2))^(1/6));
COULCUT = SQRTX/CALPHA;
CALPHA2 =   CALPHA*CALPHA;
if (COULCUT > 50)
  COULCUT = 50;
  CALPHA = SQRTX/COULCUT;
end
COULCUT2 = COULCUT*COULCUT;
CALPHA2 = CALPHA*CALPHA;

RELPERM = 1;
KECONST = 14.3996437701414*RELPERM;
TFACT  = 16.0/(5.0*KECONST);

SQRTPI = sqrt(pi);

FCOUL = zeros(3,1);
COULOMBV = 0;

TI = TFACT*U(I);
TI2 = TI*TI;
TI3 = TI2*TI;
TI4 = TI2*TI2;
TI6 = TI4*TI2;

SSA = TI;
SSB = TI3/48;
SSC = 3*TI2/16;
SSD = 11*TI/16;
SSE = 1;

Ra = [RX(I),RY(I),RZ(I)];

for nnI = 1:nrnnlist(I)
  Rb(1) = nnRx(I,nnI);
  Rb(2) = nnRy(I,nnI);
  Rb(3) = nnRz(I,nnI);
  J = nnType(I,nnI);

    Rab = Rb-Ra;  % OBS b - a !!!
    dR = norm(Rab);
    MAGR = dR;
    MAGR2 = dR*dR;

    if (dR <= COULCUT) & (dR > 1e-12)

       TJ = TFACT*U(J);
       DC = Rab/dR;

       %! Using Numerical Recipes ERFC
       Z = abs(CALPHA*MAGR);
       NUMREP_ERFC = erfc(Z);

       CA = NUMREP_ERFC/MAGR;
       COULOMBV = COULOMBV + DELTAQ(J)*CA;
       ccnt = ccnt + 1;
       TEST(ccnt) = DELTAQ(J)*CA;
       CA = CA + 2*CALPHA*exp( -CALPHA2*MAGR2 )/SQRTPI;
       FORCE = -KECONST*DELTAQ(I)*DELTAQ(J)*CA/MAGR;
       EXPTI = exp(-TI*MAGR );

       if Element_Type(I) == Element_Type(J)
           COULOMBV = COULOMBV - DELTAQ(J)*EXPTI*(SSB*MAGR2 + SSC*MAGR + SSD + SSE/MAGR);
       ccnt = ccnt + 1;
       TEST(ccnt) = - DELTAQ(J)*EXPTI*(SSB*MAGR2 + SSC*MAGR + SSD + SSE/MAGR);
           FORCE = FORCE + (KECONST*DELTAQ(I)*DELTAQ(J)*EXPTI)*((SSE/MAGR2 - 2*SSB*MAGR - SSC) + SSA*(SSB*MAGR2 + SSC*MAGR + SSD + SSE/MAGR));
       else
           TJ2 = TJ*TJ;
           TJ3 = TJ2*TJ;
           TJ4 = TJ2*TJ2;
           TJ6 = TJ4*TJ2;
           EXPTJ = exp( -TJ*MAGR );
           TI2MTJ2 = TI2 - TJ2;
           TJ2MTI2 = -TI2MTJ2;
           SA = TI;
           SB = TJ4*TI/(2 * TI2MTJ2 * TI2MTJ2);
           SC = (TJ6 - 3*TJ4*TI2)/(TI2MTJ2 * TI2MTJ2 * TI2MTJ2);
           SD = TJ;
           SE = TI4*TJ/(2 * TJ2MTI2 * TJ2MTI2);
           SF = (TI6 - 3*TI4*TJ2)/(TJ2MTI2 * TJ2MTI2 * TJ2MTI2);

           COULOMBV = COULOMBV - (DELTAQ(J)*(EXPTI*(SB - (SC/MAGR)) + EXPTJ*(SE - (SF/MAGR))));
           FORCE = FORCE + KECONST*DELTAQ(I)*DELTAQ(J)*((EXPTI*(SA*(SB - (SC/MAGR)) - (SC/MAGR2))) + (EXPTJ*(SD*(SE - (SF/MAGR)) - (SF/MAGR2))));
       end

       FCOUL(1) = FCOUL(1) + DC(1)*FORCE;
       FCOUL(2) = FCOUL(2) + DC(2)*FORCE;
       FCOUL(3) = FCOUL(3) + DC(3)*FORCE;
    end
end
COULOMBV = KECONST * COULOMBV;

