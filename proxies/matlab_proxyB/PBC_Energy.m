%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Total energy calculation %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Etot,Eband0,Ecoul,Edipole,S_ent] = PBC_Energy(H0,Efield,D0,C,D,q,R,Te);

%% E = 2*trace(H0*(D-D0)) + 0.5*sum_ij{i!=j} (qi*Cij*qj) + 0.5*sum_i qi^2*Ui - Efield*mu
%% dipole = mu(:) = sum_i qi*R(i,:); qi = 2*(D_ii-D0_ii)

kB = 8.61739e-5; % eV/K;
N = max(size(q));
Eband0 = 2*trace(H0*(D-D0));  % Single-particle/band energy

Ecoul = 0.5*q'*C*q;           % Coulomb energy

Edipole = 0;
for i = 1:N
  Edipole = Edipole - q(i)*R(i)*Efield;  % External-field-Dipole interaction energy
end

f = eig(D);
S_ent = 0; eps = 1e-9
for i = 1:N
  if (f(i) < 1-eps) & (f(i) > eps)
    S_ent = - kB*(f(i)*log(f(i)) + (1-f(i))*log(1-f(i)));
  end
end

Etot = Eband0 + Ecoul + Edipole - 2*Te*S_ent;         % Total energy


