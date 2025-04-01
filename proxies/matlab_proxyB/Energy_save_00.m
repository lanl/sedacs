%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Total energy calculation %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Etot,Eband0,Ecoul,Edipole,S_ent] = Energy(H0,U,Efield,D0,C,D,q,Rx,Ry,Rz,f,Te);

%% E = 2*trace(H0*(D-D0)) + 0.5*sum_ij{i!=j} (qi*Cij*qj) + 0.5*sum_i qi^2*Ui - Efield*mu - 2*Te*S_ent
%% dipole = mu(:) = sum_i qi*R(i,:); qi = 2*(D_ii-D0_ii)


kB = 8.61739e-5; % eV/K;
N = max(size(q));
D0 = diag(D0);
Eband0 = 2*trace(H0*(D-D0));  % Single-particle/band energy

Ecoul = 0.5*q'*C*q;           % Coulomb energy
for i = 1:N
  Ecoul = Ecoul + 0.5*q(i)*U(i)*q(i);
end

Edipole = 0;
for i = 1:N
  Edipole = Edipole - q(i)*(Rx(i)*Efield(1)+Ry(i)*Efield(2)+Rz(i)*Efield(3));  % External-field-Dipole interaction energy
end

S_ent = 0; eps = 1e-9
for i = 1:N
  if (f(i) < 1-eps) & (f(i) > eps)
    S_ent = - kB*(f(i)*log(f(i)) + (1-f(i))*log(1-f(i)));
  end
end

Etot = Eband0 + Ecoul + Edipole - 2*Te*S_ent;         % Total energy


