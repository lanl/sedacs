function x0 = Slater_Koster_Block(Ra,Rb,LBox,Type_pair,fss_sigma,fsp_sigma,fps_sigma,fpp_sigma,fpp_pi,diagonal)

%%% Standard Slater-Koster sp-parameterization for an atomic block between a pair of atoms
%%% IDim, JDim: dimensions of the output block, e.g. 1 x 4 for H-O or 4 x 4 for O-O, or 4 x 1 for O-H
%%% Ra, Rb: are the vectors of the positions of the two atoms
%%% LBox: Periodic boundary conditions, i.e. length of box in x, y, z (cubic box only)
%%% Type_pair(1 or 2): Character of the type of each atom in the pair, e.g. 'H' for hydrogen of 'O' for oxygen
%%% fss_sigma, ... , fpp_pi: paramters for the bond integrals
%%% diagonal(1 or 2): atomic energies Es and Ep or diagonal elements of the overlap i.e. diagonal = 1


atom_type_a = Type_pair(1); atom_type_b = Type_pair(2);
x0 = zeros(4);
RXb = Rb(1); RYb = Rb(2); RZb = Rb(3);

  Rab = Rb-Ra;  % OBS b - a !!!
  dR = norm(Rab);

  if dR < 1e-10 % same position and thus the same type atom_type_a = atom_type_b, Ra = Rb
    if atom_type_a == 'H'       % s atom 1 x 1
        x0(1,1) = x0(1,1) + diagonal(1);  % diagonal(1) = Es for atoms, = 1 for overlap
    else                        % sp atom 4 x 4 Diagonal Only
        x0(1,1) = x0(1,1) + diagonal(1);  % diagonal(1) = Es for atoms, = 1 for overlap
        x0(2,2) = x0(2,2) + diagonal(2);  % diagonal(2) = Ep for atoms, = 1 for overlap
        x0(3,3) = x0(3,3) + diagonal(2);
        x0(4,4) = x0(4,4) + diagonal(2);
    end
  else
    L = Rab(1)/dR;   % Direction cosines
    M = Rab(2)/dR;
    N = Rab(3)/dR;

    if atom_type_a == 'H'
      if atom_type_b == 'H'  % s-s  overlap 1 x 1 block
        HSSS = BondIntegral(dR,fss_sigma); % Calculate the s-s bond integral
        x0(1,1) = x0(1,1) + HSSS;
      else              % s-sp overlap 1 x 4 block
        HSSS = BondIntegral(dR,fss_sigma);
        HSPS = BondIntegral(dR,fsp_sigma);
        x0(1,1) = x0(1,1) + HSSS;
        x0(1,2) = x0(1,2) + L*HSPS;
        x0(1,3) = x0(1,3) + M*HSPS;
        x0(1,4) = x0(1,4) + N*HSPS;
      end
    else
      if atom_type_b == 'H'  % sp-s overlap 4 x 1 block
        HSSS = BondIntegral(dR,fss_sigma);
        HSPS = BondIntegral(dR,fsp_sigma);
        x0(1,1) = x0(1,1) + HSSS;
        x0(2,1) = x0(2,1) - L*HSPS;
        x0(3,1) = x0(3,1) - M*HSPS;
        x0(4,1) = x0(4,1) - N*HSPS;
      else              % sp-sp overlap
        HSSS = BondIntegral(dR,fss_sigma);
        HSPS = BondIntegral(dR,fsp_sigma);
        HPSS = BondIntegral(dR,fps_sigma);
        HPPS = BondIntegral(dR,fpp_sigma);
        HPPP = BondIntegral(dR,fpp_pi);

        PPSMPP = HPPS - HPPP;
        PXPX = HPPP + L*L*PPSMPP;
        PXPY = L*M*PPSMPP;
        PXPZ = L*N*PPSMPP;
        PYPX = M*L*PPSMPP;
        PYPY = HPPP + M*M*PPSMPP;
        PYPZ = M*N*PPSMPP;
        PZPX = N*L*PPSMPP;
        PZPY = N*M*PPSMPP;
        PZPZ = HPPP + N*N*PPSMPP;

        x0(1,1) = x0(1,1) + HSSS;
        x0(1,2) = x0(1,2) + L*HSPS; % or 0
        x0(1,3) = x0(1,3) + M*HSPS; % or 0
        x0(1,4) = x0(1,4) + N*HSPS; % or 0

        x0(2,1) = x0(2,1) - L*HPSS; % or 0

        x0(2,2) = x0(2,2) + PXPX;
        x0(2,3) = x0(2,3) + PXPY;
        x0(2,4) = x0(2,4) + PXPZ;

        x0(3,1) = x0(3,1) - M*HPSS; % or 0

        x0(3,2) = x0(3,2) + PYPX;
        x0(3,3) = x0(3,3) + PYPY;
        x0(3,4) = x0(3,4) + PYPZ;
        x0(4,1) = x0(4,1) - N*HPSS; % or 0
        x0(4,2) = x0(4,2) + PZPX;
        x0(4,3) = x0(4,3) + PZPY;
        x0(4,4) = x0(4,4) + PZPZ;
      end
    end
  end

