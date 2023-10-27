program proxy_a
  use proxy_a_mod
  implicit none
  real(dp),allocatable :: coords(:,:)
  integer,allocatable :: types(:)
  real(dp), allocatable :: H(:,:)
  integer :: nats

  nats = 10 
  call get_random_coordinates(nats,coords)
  allocate(types(nats)); types = 1
  allocate(H(nats,nats))
  call get_hamiltonian(coords,types,H,.true.)
  write(*,*)"Hamiltonian matrix =",H

end program proxy_a

