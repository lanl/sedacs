%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simple N^2 brute force nearest neighborlist %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(Rx,Ry,Rz,LBox,Rcut,N)

% Rx, Ry, Rz are the coordinates of atoms
% LBox dimensions of peridic BC
% N number of atoms
% nrnnlist(I): number of atoms within distance of Rcut from atom I including atoms in the skin
% nndist(I,J): distance between atom I(in box) and J (including atoms in the skin)
% nnRx(I,J): x-coordinte of neighbor J to I within RCut (including atoms in the skin)
% nnRy(I,J): y-coordinte of neighbor J to I within RCut (including atoms in the skin)
% nnRz(I,J): z-coordinte of neighbor J to I within RCut (including atoms in the skin)
% nnType(I,J): The neighbor J of I corresponds to some translated atom number in the box that we need to keep track of
% nnStruct(I,J): The neigbors J to I within Rcut that are all within the box (not in the skin).
% nrnnStruct(I): Number of neigbors to I within Rcut that are all within the box (not in the skin).

Lx = LBox(1); Ly = LBox(2); Lz = LBox(3); % Dimensions of periodic BC
nx = floor(Lx/Rcut); ny = floor(Ly/Rcut); nz = floor(Lz/Rcut); % Division into # cell boxes: nx, ny, nz

nndist = zeros(N,floor(10+4*N/((nx+2)*(ny+2)*(nz+2)))); % Allocation of memory, not optimized!
nnRx = zeros(N,floor(10+4*N/((nx+2)*(ny+2)*(nz+2))));   % Make sure the fastes allocation for Forstran is used, e.g. (N,1) instead of (1,N) or the opposite!
nnRy = zeros(N,floor(10+4*N/((nx+2)*(ny+2)*(nz+2))));
nnRz = zeros(N,floor(10+4*N/((nx+2)*(ny+2)*(nz+2))));
type = zeros(10*N);
nnType = zeros(N,floor(10+4*N/((nx+2)*(ny+2)*(nz+2))));
nnStruct = zeros(N,floor(10+4*N/((nx+2)*(ny+2)*(nz+2))));
nrnnStruct = zeros(N,1);
nrnnlist = zeros(N,1);

% Simple N^2 brute force nearest neighborlist
 for i = 1:N
   cnt = 0;
   tmp = zeros(N,1);
   for m = 1:N
     for j = -1:1
     for k = -1:1
     for l = -1:1
       Tx = Rx(m)+j*Lx;  % Search all neigbors within a single translation (multiple translations could be necessary for small systems!
       Ty = Ry(m)+k*Ly;
       Tz = Rz(m)+l*Lz;
       dist = norm([Rx(i),Ry(i),Rz(i)]-[Tx,Ty,Tz]);
%       if (dist < Rcut) & (dist > 1e-12)  % Neighbors within Rcut inlcuidng translated atoms in the "skin"
       if (dist < Rcut)   % Neighbors within Rcut inlcuidng translated atoms in the "skin"
         cnt = cnt + 1;
         nndist(i,cnt) = dist;
         nnRx(i,cnt) = Tx;
         nnRy(i,cnt) = Ty;
         nnRz(i,cnt) = Tz;
         nnType(i,cnt) = m;  % Neigbor is number of original ordering number m in the box that might have been stranslated to the skin
         tmp(m) = m;
       end
     end
     end
     end
   end
   nrnnlist(i) = cnt;
   cnt2 = 0;
   for ss = 1:N
     if tmp(ss) > 0 % Includes only neighbors in the box within Rcut (without the skin)
       cnt2 = cnt2 + 1;
       nnStruct(i,cnt2) = ss;
     end
   end
   nrnnStruct(i) = cnt2;
 end

