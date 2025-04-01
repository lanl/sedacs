  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function [P0,P1,mu0,mu1] = DM_PRT_Fermi(H0,H1,T,mu_0,mu1,Ne,m,eps,MaxIt)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  N = max(size(H0));
  I = eye(N); mu_1_start = mu1;
  mu0 = mu_0; mu1 = mu1; % Intial guess
  OccErr = 1;
  Cnt = 0;
  while OccErr > eps
    %kB = 6.33366256e-6; % Ry/K;
    %kB = 3.166811429e-6; % Ha/K;
    kB = 8.61739e-5; % eV/K;
    beta = 1/(kB*T);    % Temp in Kelvin
    cnst = 2^(-2-m)*beta;
    P0 = 0.5*I - cnst*(H0-mu0*I);
    P1 = -cnst*(H1-mu1*I);
    for i = 1:m
  %    hh = sort(real(eig(P0)));
      P02 = P0*P0;
      DX1 = P0*P1+P1*P0;
      ID0 = inv(2*(P02-P0) + I);
      P_0 = ID0*P02;
      P_1 = ID0*(DX1+2*(P1-DX1)*P_0);
      P0 = P_0; P1 = P_1;
    end
    TrdPdmu = trace(beta*P0*(I-P0));
    if abs(TrdPdmu) > 1e-9
      mu0 = mu0 + (Ne - trace(P0))/TrdPdmu;
      mu1 = mu1 + (0  - trace(P1))/TrdPdmu;
      OccErr = abs(trace(P0+P1)-Ne);
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
  P1 = P1 - trace(P1)*I/N;
