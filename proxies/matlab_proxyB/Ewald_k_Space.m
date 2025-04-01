function [COULOMBV,FCOUL] = Ewald_k_Space(RX,RY,RZ,LBox,DELTAQ,Nr_atoms,COULACC,TIMERATIO)

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
KCUTOFF = 2*CALPHA*SQRTX;
KCUTOFF2 = KCUTOFF*KCUTOFF;
CALPHA2 = CALPHA*CALPHA;
FOURCALPHA2 = 4*CALPHA2;
RECIPVECS = zeros(3,3);
RECIPVECS(1,1) = 2*pi/LBox(1);
RECIPVECS(2,2) = 2*pi/LBox(2);
RECIPVECS(3,3) = 2*pi/LBox(3);
LMAX = floor(KCUTOFF / sqrt(RECIPVECS(1,1)*RECIPVECS(1,1)));
MMAX = floor(KCUTOFF / sqrt(RECIPVECS(2,2)*RECIPVECS(2,2)));
NMAX = floor(KCUTOFF / sqrt(RECIPVECS(3,3)*RECIPVECS(3,3)));

RELPERM = 1;
KECONST = 14.3996437701414*RELPERM;

SQRTPI = sqrt(pi);

FCOUL = zeros(3,Nr_atoms);
COULOMBV = zeros(1,Nr_atoms);
SINLIST = zeros(Nr_atoms);
COSLIST = zeros(Nr_atoms);
%CR = R';

for L = 0:LMAX

  if L == 0
    MMIN = 0;
  else
    MMIN = -MMAX;
  end

  L11 = L*RECIPVECS(1,1);
  L12 = L*RECIPVECS(1,2);
  L13 = L*RECIPVECS(1,3);

  for M = MMIN:MMAX

        NMIN = -NMAX;

        if (L == 0) &  (M == 0)
          NMIN = 1;
        end

        M21 = L11 + M*RECIPVECS(2,1);
        M22 = L12 + M*RECIPVECS(2,2);
        M23 = L13 + M*RECIPVECS(2,3);

        for N = NMIN:NMAX
           K(1) = M21 + N*RECIPVECS(3,1);
           K(2) = M22 + N*RECIPVECS(3,2);
           K(3) = M23 + N*RECIPVECS(3,3);
           K2 = K(1)*K(1) + K(2)*K(2) + K(3)*K(3);
           if K2 <= KCUTOFF2
              PREFACTOR = 8*pi*exp(-K2/(4*CALPHA2))/(COULVOL*K2);
              PREVIR = (2/K2) + (2/(4*CALPHA2));

              COSSUM = 0;
              SINSUM = 0;

              %! Doing the sin and cos sums

              for I = 1:Nr_atoms
                 %DOT = K(1)*CR(1,I) + K(2)*CR(2,I) + K(3)*CR(3,I);
                 DOT = K(1)*RX(I) + K(2)*RY(I) + K(3)*RZ(I);
                 %! We re-use these in the next loop...
                 SINLIST(I) = sin(DOT);
                 COSLIST(I) = cos(DOT);
                 COSSUM = COSSUM + DELTAQ(I)*COSLIST(I);
                 SINSUM = SINSUM + DELTAQ(I)*SINLIST(I);
              end
              COSSUM2 = COSSUM*COSSUM;
              SINSUM2 = SINSUM*SINSUM;

              %! Add up energy and force contributions

              KEPREF = KECONST*PREFACTOR;
              for I = 1:Nr_atoms
                 COULOMBV(I) = COULOMBV(I) + KEPREF*(COSLIST(I)*COSSUM + SINLIST(I)*SINSUM);
                 FORCE = KEPREF * DELTAQ(I)*(SINLIST(I)*COSSUM - COSLIST(I)*SINSUM);
                 FCOUL(1,I) = FCOUL(1,I) + FORCE*K(1);
                 FCOUL(2,I) = FCOUL(2,I) + FORCE*K(2);
                 FCOUL(3,I) = FCOUL(3,I) + FORCE*K(3);
              end

              KEPREF = KEPREF * (COSSUM2 + SINSUM2);

           end
        end
  end
end

%! Point self energy
CORRFACT = 2*KECONST*CALPHA/SQRTPI;
COULOMBV = COULOMBV - CORRFACT*DELTAQ;

