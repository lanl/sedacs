function h = BondIntegral(dR,f)

%%% dR: distance between atoms
%%% f: paramters/coefficeints for the bond integral

  if (dR <= f(7))
    RMOD = dR - f(6);
    POLYNOM = RMOD*(f(2) + RMOD*(f(3) + RMOD*(f(4) + f(5)*RMOD)));
    X = exp(POLYNOM);
  elseif (dR > f(7)) & (dR < f(8))
    RMINUSR1 = dR - f(7);
    X = f(9) + RMINUSR1*(f(10) + RMINUSR1*(f(11) + RMINUSR1*(f(12) + RMINUSR1*(f(13) + RMINUSR1*f(14)))));
  else
    X = 0;
  end
  h = f(1)*X;


