function [H,Hcoul,Hdipole,D,q] = PBC_SCF(H0,Efield,D0,C,q,R,nocc,Te)

N = max(size(H0));
q_new = q; q_old = 0*q;
it = 0; res = 1;
Vdipole = zeros(N,1);
h = sort(eig(H0));
mu0 = 0.5*(h(nocc)+h(nocc+1)); mu1 = 0;
while res > 1e-10
 it = it + 1;
 Hdipole = diag(-R*Efield);
 Hcoul = diag(C*q_new);
 H = H0 + Hcoul + Hdipole;
% DA = DensityMatrix(H,nocc);  % 2*trace(D) = Ne = 2*nocc
 [D,mu0] = DM_Fermi(H,Te,mu0,nocc,16,1e-9,50);
 q_old = q_new;
 q = 2*diag(D-D0);
 q_new = 0.2*q + (1-0.2)*q_old;
 res = norm(q-q_old);
end

