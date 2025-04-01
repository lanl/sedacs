function A = ScaleTail(A)
if abs(A(1)) < 1e-12
  A(9:14) = 0;
else
  R1 = A(7);
  RCUT = A(8);
  R1SQ = R1*R1;
  RMOD = R1 - A(6);
  POLYNOM = RMOD*(A(2) + RMOD*(A(3) + RMOD*(A(4) + A(5)*RMOD)));
  SCL_R1 = exp(POLYNOM);
  DELTA = RCUT - R1;
%! Now we're using a 6th order polynomial: fitted to value, first,
%! and second derivatives at R1 and R_cut
  A(9) = SCL_R1;
  RMOD = R1 - A(6);
  DPOLY = A(2) + 2*A(3)*RMOD + 3*A(4)*RMOD*RMOD + 4*A(5)*RMOD*RMOD*RMOD;
  A(10) = DPOLY*SCL_R1;
  DDPOLY = 2*A(3) + 6*A(4)*RMOD + 12*A(5)*RMOD*RMOD;
  A(11) = (DPOLY*DPOLY + DDPOLY)*SCL_R1/2;
  DELTA2 = DELTA*DELTA;
  DELTA3 = DELTA2*DELTA;
  DELTA4 = DELTA3*DELTA;
  A(12) = (-1/DELTA3)*(3*A(11)*DELTA2 + 6*A(10)*DELTA + 10*A(9));
  A(13) = (1/DELTA4)*(3*A(11)*DELTA2 + 8*A(10)*DELTA + 15*A(9));
  A(14) = (-1/(10*DELTA3))*(6*A(13)*DELTA2 + 3*A(12)*DELTA + A(11));
end
