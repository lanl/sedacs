program proxy_a
  use proxy_a_mod
  implicit none
  real(dp),allocatable :: coords(:,:)
  integer,allocatable :: types(:)
  real(dp), allocatable :: H(:,:),D(:,:)
  integer :: nats,nocc,i,j

  nats = 2
 
  call get_random_coordinates(nats,coords)
  do i = 1,nats
  write(*,*)coords(1,i)
  enddo
  allocate(types(nats)); types = 1
  allocate(H(nats,nats))
  call get_hamiltonian(coords,types,H,.true.)
  write(*,*)"Hamiltonian matrix"
  do i=1,nats
    do j=1,nats
     write(*,*)H(i,j)
    enddo
  enddo

  allocate(D(nats,nats))
  nocc = int(real(nats,dp)/2.0_dp)
  call get_densityMatrix(H,nocc,D,.true.)
  write(*,*)"Density matrix=",D

end program proxy_a

