function [q] = Get_q(D,S,H_Index_Start,H_Index_End,Znuc,Nats)

Z = S^(-1/2);
DS = 2*diag(D*S);
q = zeros(Nats,1);
for i = 1:Nats
  q(i) = sum(DS(H_Index_Start(i):H_Index_End(i))) - Znuc(i);
end

