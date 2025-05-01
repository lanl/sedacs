%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCF linear response calculation with respect to perturbation H1       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [H,H_1,Hcoul,Hdipole,D,D1,q] = PBC_SCF_PRT_0(H0,H1,C,Efield,D0,U,q,R,nocc)

N = max(size(H0));
q_new = q; q_old = 0*q;                    % Initial guess
q1 = 0*q; q1_new = q1; q1_old = q1;        % Initial guess
Vdipole = zeros(N,1);

it = 0;
while norm(q-q_old)+norm(q1-q1_old) > 1e-9 % Continue until convergence
 it = it + 1;

 V1coul = C*q1_new;          % Linear response in Coulomb/Hartree potential

 Vcoul = C*q_new;            % Coulomb/Hartree potential
 Hcoul = diag(Vcoul);        % Hamiltonian from the linear response in Coulomb/Hartree potential
 Hdipole = diag(-R*Efield);  % Hamiltonian from external field dipole interaction

 H_1 = H1 + diag(V1coul);    % Total net linear response Hamiltonian
 H = H0 + Hcoul + Hdipole;   % Total 0th-order Hamiltonian
 [D,D1] = DensityMatrixPRT(H,H_1,nocc);  % 2*trace(D) = Ne = 2*nocc, D = density matrix, D1 = response density matrix
% OccErr = trace(D)-nocc
% OccErr1 = trace(D1)-0
% IdErr = norm(D*D-D)
% IdErr1 = norm(D1*D+D*D1-D1)
% ComErr = norm(D*H-H*D)
% ComErr1 = norm(D1*H-H*D1 + D*H_1-H_1*D)
% pause



 q1_old = q1_new;
 q1 = 2*diag(D1); 
 q1_new = 0.1*q1 + (1-0.1)*q1_old;  % Simple linear mixing

 q_old = q_new;
 q = 2*diag(D-D0);
 q_new = 0.1*q + (1-0.1)*q_old;     % Simple linear mixing
end

