%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Detects new and removed edges between two graphs G1 and G2 %%
%%         The routine does not require ordering              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
N = 10; M = 5;  % N nodes and max M edges

% Create first graph randomly
G1 = zeros(N,M); 
NNZ1 = floor(1 + M*rand(10,1));  % [1,M] for each node
TIO = [1:N]';
for i = 1:N
  for k = 1:7  % reshuffle ordering
    a = floor(1 + N*rand(1));
    b = floor(1 + N*rand(1));
    tmp = TIO(a); TIO(a) = TIO(b); TIO(b) = tmp;
  end
  for j = 1:NNZ1(i)
    G1(i,j) = TIO(j);
  end
end

% Create second graph randomly
G2 = zeros(N,M); 
NNZ2 = floor(1 + M*rand(10,1));  % [1,M] for each node
TIO = [1:N]';
for i = 1:N
  for k = 1:7  % reshuffle ordering
    a = floor(1 + N*rand(1));
    b = floor(1 + N*rand(1));
    tmp = TIO(a); TIO(a) = TIO(b); TIO(b) = tmp;
  end
  for j = 1:NNZ2(i)
    G2(i,j) = TIO(j);
  end
end

% Analyze the difference from G1 to G2

%  Added edges
G_added = zeros(N,M); N_added = zeros(N,1);
v = zeros(1,N);
for i = 1:N
  for j = 1:NNZ1(i)
    v(G1(i,j)) = 1;
  end
  k = 0;
  for j = 1:NNZ2(i)
    if (v(G2(i,j)) == 0)
      k = k + 1;
      G_added(i,k) = G2(i,j);
    end
  end
  N_added(i) = k;  % Number of added edges for each vertex i
  v(G1(i,1:NNZ1(i))) = 0;
  v(G2(i,1:NNZ2(i))) = 0;
end

% Removed edges
G_removed = zeros(N,M); N_removed = zeros(N,1);
v = zeros(1,N);
for i = 1:N
  for j = 1:NNZ2(i)
    v(G2(i,j)) = 1;
  end
  k = 0;
  for j = 1:NNZ1(i)
    if (v(G1(i,j)) == 0)
      k = k + 1;
      G_removed(i,k) = G1(i,j);
    end
  end
  N_removed(i) = k; % Number of removed edges for each vertex i
  v(G1(i,1:NNZ1(i))) = 0;
  v(G2(i,1:NNZ2(i))) = 0;
end

