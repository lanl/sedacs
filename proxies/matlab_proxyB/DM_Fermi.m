  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function [P0,mu0] = DM_Fermi(H0,T,mu_0,Ne,m,eps,MaxIt)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  N = max(size(H0));
  I = eye(N); 
  mu0 = mu_0; 
  OccErr = 1;
  Cnt = 0;
  while OccErr > eps
    %kB = 6.33366256e-6; % Ry/K;
    %kB = 3.166811429e-6; % Ha/K;
    kB = 8.61739e-5; % eV/K;
    beta = 1/(kB*T);    % Temp in Kelvin
    cnst = 2^(-2-m)*beta;
    P0 = 0.5*I - cnst*(H0-mu0*I);
    for i = 1:m
      P02 = P0*P0;
      ID0 = inv(2*(P02-P0) + I);
      P_0 = ID0*P02;
      P0 = P_0; 
    end
    TrdPdmu = trace(beta*P0*(I-P0));
    if abs(TrdPdmu) > 1e-8
      mu0 = mu0 + (Ne - trace(P0))/TrdPdmu;
      OccErr = abs(trace(P0)-Ne);
    else
      OccErr = 0;
    end
    Cnt = Cnt + 1;
    if (Cnt >= MaxIt)
      OccErr;
      OccErr = 0;
      Cnt = MaxIt;
    end
  end
  % Adjust occupation
  P0 = P0 + ((Ne - trace(P0))/TrdPdmu)*beta*P0*(I-P0);
