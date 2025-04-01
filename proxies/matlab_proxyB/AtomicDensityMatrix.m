function [D_atomic] = AtomicDensityMatrix(Nr_atoms,H_INDEX_START,H_INDEX_END,HDIM,Znuc);

D_atomic = zeros(1,HDIM);

INDEX = 0;
for I = 1:Nr_atoms
  N_orb = H_INDEX_END(I)-H_INDEX_START(I) + 1;
  if N_orb == 1
    INDEX = INDEX + 1;
    D_atomic(INDEX) = Znuc(I);
  else
    if Znuc(I) <= 2
      INDEX = INDEX + 1;
      D_atomic(INDEX) = Znuc(I);

      INDEX = INDEX + 1;
      D_atomic(INDEX) = 0;
      INDEX = INDEX + 1;
      D_atomic(INDEX) = 0;
      INDEX = INDEX + 1;
      D_atomic(INDEX) = 0;

    else
      INDEX = INDEX + 1;
      D_atomic(INDEX) = 2;

      INDEX = INDEX + 1;
      OCC = (Znuc(I)-2)/3;
      D_atomic(INDEX) = OCC;
      INDEX = INDEX + 1;
      D_atomic(INDEX) = OCC;
      INDEX = INDEX + 1;
      D_atomic(INDEX) = OCC;
    end
  end
end
