clear;

N = 260; Nocc = 77; eps = 1e-3; decay_min = 0.1;
H = zeros(N);

m = 78; a = 3.817632; c = 0.816371; x = 1.029769;
n = 13; b = 1.927947; d = 3.386142; y = 2.135545;
% Construct quasi-randomized Hamiltonian H for the full system
cnt = 0; R = zeros(N,1);
for i = 1:N
  x = mod(a*x+c,m); y = mod(b*y+d,n);  % Quasi-coordinates and atoms types!
  xx(i) = x; yy(i) = y; R(i) = i;
  for j = i:N
     R(j) = j;
     cnt = cnt + 1;
     tmp = (x/m)*exp(-(y/n + decay_min)*(R(i)-R(j))^2);
     H(i,j) = tmp; H(j,i) = tmp;
  end
end

% Get some estimate of the chemical potential
[Q,E] = eig(H);
h = sort(diag(E));
mu0 = 0.5*(h(Nocc + 1) + h(Nocc));
%D = zeros(N);
%for i = 1:N
%  if E(i,i) < mu0
%     D = D + Q(:,i)*Q(:,i)';
%  end
%end
%D = Thresh(D,eps);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we will do the same using graph-partitioning %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Estimate Connectivity Graph
G_H = Thresh(H,1e-6);
G_H = Adjacency(G_H);  % 1s or 0s
G = G_H;

% Chose core partitioning, should be flexible!

n_core = 4;  % N needs to be divisable by n_core!
core_part = zeros(N,N/n_core);
k = 0;
for i = 1:n_core:N
   k = k + 1;
   core_part(i:i+n_core-1,k) = 1;
end
nr_core_partitionings = k;

% Extract Core + Halo partitionings & Identify Subgraphs
core_halo_part = zeros(N,N/n_core);
SubSyst_X = zeros(N,nr_core_partitionings);         % N should be estimated and be smaller
SubSyst_Y = zeros(N,nr_core_partitionings);         % N should be estimated and be smaller
SubSyst_R = zeros(N,nr_core_partitionings);         % N should be estimated and be smaller
SubSyst_C = zeros(N,nr_core_partitionings);         % N should be estimated and be smaller
SubSyst_CH = zeros(N,nr_core_partitionings);        % N should be estimated and be smaller
SubSyst_C_index = zeros(N,nr_core_partitionings);   % N should be estimated and be smaller
SubSyst_nnz = zeros(nr_core_partitionings,1);       % Keeps track of the number of core + halo atoms



% Main outer loop updating the estimates of the data dependency graph (needed for the first SCF)
for Graph_Extension = 1:8

  for k = 1:nr_core_partitionings
     % Double jump-like estimate from G and G_H (sparse matrix-vector mult)
%     core_halo_part(:,k) = G*(G_H*(G*core_part(:,k)));           % Rapid growth, best for initial SCF
%     core_halo_part(:,k) = G_H*(G*(G_H*core_part(:,k)));         % Slower growth better for QMD
    core_halo_part(:,k) = G*(G_H*core_part(:,k));     % Even slower growth and even better for QMD (but nonsymmetric!)
%    core_halo_part(:,k) = G*(G_H*core_part(:,k)) + G_H*(G*core_part(:,k));   % Slower growth for QMD (symmetric!)
     
  
     % Identify subgraph indices and their "coordinates and atom types"
     cnt = 0; cnt_core = 0;
     for i = 1:N
       if core_halo_part(i,k) > 0
          cnt = cnt + 1;
          SubSyst_X(cnt,k) =  xx(i);
          SubSyst_Y(cnt,k) =  yy(i);
          SubSyst_R(cnt,k) =  R(i);
          SubSyst_CH(cnt,k) = i;           % Keep track of the atoms of the full system belonging to the core + halo
         if core_part(i,k) > 0
           cnt_core = cnt_core + 1;             %% These names "C and C_index" could probably be better!
           SubSyst_C(cnt_core,k) = i;           % Keep track of the atoms of the full system belonging to the core only
           SubSyst_C_index(cnt_core,k) = cnt;   % Keep track of the atoms of the subgraph belonging to the core
         end
       end
     end
     SubSyst_nnz(k) = cnt;  % Number of core + halo elements for subgraph partitioning k
  end
  core_halo_part = Adjacency(core_halo_part);  % 1s or 0s
  spy(core_halo_part)
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% MAIN PARALLEL LOOP  (here only sequential!) %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  E_occ = zeros(N,k);          % Should be SubSyst_nnz(k) instead of N
  E_val = zeros(N,k);          % Should be SubSyst_nnz(k) instead of N
  for k = 1:nr_core_partitionings  % Here nr_core_partitionings = N/n_core   % Should be distributed over separate nodes or threads
  
     % For each subgraph paritioning k build Subgraph Hamiltonian of core + halo using SubSyst information only
     H_SubSyst = zeros(SubSyst_nnz(k));
     for i = 1:SubSyst_nnz(k)
       x = SubSyst_X(i,k);
       y = SubSyst_Y(i,k);
       Ri = SubSyst_R(i,k);
       for j = i:SubSyst_nnz(k)
         Rj = SubSyst_R(j,k);
         tmp  = (x/m)*exp(-(y/n + decay_min)*(Ri-Rj)^2);
         H_SubSyst(i,j) =  tmp;
         H_SubSyst(j,i) = tmp;
       end
     end
     % Diagonalize subgraph Hamiltonian
     [Q,E] = eig(H_SubSyst);
     for i = 1:SubSyst_nnz(k)     % i runs over all eigenstates of the subgraph system
       E_val(i,k) = E(i,i);       % Eigenvalue i from subgraph system k
       for j = 1:n_core           % Assumes number of cores are the same for each subgraph (not a general solution)
         % Occupation contribution to the collected full system from each eigeinstate i (core parts only) of subsystem k
         E_occ(i,k) = E_occ(i,k) + Q(SubSyst_C_index(j,k),i)^2;
       end
     end
     
  end
  
  %% Determine the Chemical Potential for the Fermi-operator expansion assuming Te > 0 using Newton Raphson
  %% Bases of shared information: E_val(i,k), and E_occ(i,k) with one pair of vectors from each subsystem
  kB = 8.61739e-5; % (ev/K) 
  Te = 200; beta = 1/(kB*Te);
 
  %% This is collected work done either on a master node or on each node in parallel 
  mu = mu0; % Initial Guess form Te = 0
  for Newt_It = 1:5
    Occ = 0; d_Occ = 0;
    for k = 1:nr_core_partitionings 
      for i = 1:SubSyst_nnz(k)
        f_i = 1/(exp(beta*(E_val(i,k) - mu)) + 1);
        Occ = Occ + f_i*E_occ(i,k);
        d_Occ = d_Occ + beta*(f_i*(1-f_i)*E_occ(i,k));
      end
    end
    mu = mu - (Occ - Nocc)/d_Occ;  % Newton iteration
  end
  %% The converged is now needed on each separate node

  %% Back into subgraphs (Do not need to recalculate H_SubSyst, Q and E. They have been kept in memory for parallel calculations)
  D = zeros(N); Occ = 0; E_band = 0;
  for k = 1:nr_core_partitionings  % Here nr_core_partitionings = N/n_core
  
     % For each graph-paritioning k build Subgraph Hamiltonian of core + halo using SubSyst information only
     % These have already been calculated
     H_SubSyst = zeros(SubSyst_nnz(k));
     for i = 1:SubSyst_nnz(k)
       x = SubSyst_X(i,k);
       y = SubSyst_Y(i,k);
       Ri = SubSyst_R(i,k);
       for j = i:SubSyst_nnz(k)
         Rj = SubSyst_R(j,k);
         tmp  = (x/m)*exp(-(y/n + decay_min)*(Ri-Rj)^2);
         H_SubSyst(i,j) =  tmp;
         H_SubSyst(j,i) = tmp;
       end
     end
     % Diagonalize subgraph Hamiltonian
     [Q,E] = eig(H_SubSyst);      % These Eigensolutions have already been calculated
     for i = 1:SubSyst_nnz(k)     % i runs over all eigenstates of the subsystem
       E_val(i,k) = E(i,i);       % Eigenvalue i from subsystem k
       f_i = 1/(exp(beta*(E(i,i) - mu)) + 1);
       for j = 1:n_core           % Assumes number of cores are the same for each subsystem (not a general solution)
         % Occupation contribution to the collected full system from each eigeinstate i (core parts only) of subsystem k
         Occ = Occ + f_i*Q(SubSyst_C_index(j,k),i)^2;
         for ii = 1:SubSyst_nnz(k)
           % Collects the composite total density matrix from subgraph system density matrices column by column
           D(SubSyst_CH(ii,k),SubSyst_C(j,k)) = D(SubSyst_CH(ii,k),SubSyst_C(j,k)) + f_i*Q(ii,i)*Q(SubSyst_C_index(j,k),i);
         end
       end
     end
     for i = 1:SubSyst_nnz(k)
       f_i = 1/(exp(beta*(E(i,i) - mu)) + 1);
       v = H_SubSyst*Q(:,i);   % overkill, only core parts of v are needed
       for j = 1:n_core
          E_band = E_band + f_i*Q(SubSyst_C_index(j,k),i)*v(SubSyst_C_index(j,k));
       end
     end
  end  

  % Update Graph with new DM
  G = Thresh(D,eps);   % This threshold tunes the accuracy for the G*G_H and G_H*G*G_H updated connectivity graphs
  G = Adjacency(G);  % 1s or 0s
  SymmetryError = norm(D-D')

  %% Exact full solution, just to check the errors
  [Q,E] = eig(H);
  DD = zeros(N);
  for i = 1:N
    f_i = 1/(exp(beta*(E(i,i) - mu)) + 1);
    ff(i) = f_i;
    DD = DD + Q(:,i)*f_i*Q(:,i)';
  end
  D_Error = norm(D-DD)
  E_Error = trace(DD*H)-E_band
  pause

end
D_D = Thresh(DD,1e-6);

