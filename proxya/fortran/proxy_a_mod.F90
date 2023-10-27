!!> proxy code a
!! A prototype engine that:
!!    - Reads the total number of atoms 
!!    - Constructs a set of random coordinates 
!!    - Constructs a simple Hamiltonian 
!!    - Computes the Density matrix from the Hamiltonian
!!
module proxy_a_mod

  implicit none 

  integer, parameter :: dp = kind(1.0D0) !Precision
  public :: get_random_coordinates

  contains

  !!> Generating random coordinates 
  !! @brief Creates a system of size "nats = Number of atoms" with coordindates having 
  !! a random (-1,1) displacement from a simple cubic lattice with parameter 2.0 Ang.
  !!
  !! @param nats The total number of atoms
  !! @return coordinates Position for every atom. z-coordinate of atom 1 = coords[0,2]
  !!
  subroutine get_random_coordinates(nats,coords)
    implicit none 
    integer :: lenght,nats,ssize,length,atomsCounter
    integer :: i,j,k
    integer, allocatable :: seedin(:)
    real(dp), allocatable :: coords(:,:)
    real(dp) :: rnd,latticeParam
    !Get random coordinates
    length = int(real(nats)**(1.0/3.0)) + 1
    allocate(coords(3,nats))
    latticeParam = 2.0
    atomsCounter = 0
    call random_seed()
    call random_seed(size=ssize)
    allocate(seedin(ssize))
    seedin = 123
    call random_seed(PUT=seedin)
    do i = 1,length
      do j = 1,length
        do k = 1,length
          atomsCounter = atomsCounter + 1
          if(atomsCounter > nats) exit
          call random_number(rnd)
          rnd = 2.0_dp*rnd - 1.0
          coords(1,atomsCounter) = i*latticeParam + rnd
          call random_number(rnd)
          rnd = 2.0_dp*rnd - 1.0
          coords(2,atomsCounter) = j*latticeParam + rnd
          call random_number(rnd)
          rnd = 2.0_dp*rnd - 1.0
          coords(3,atomsCounter) = k*latticeParam + rnd
        enddo
      enddo
    enddo
    return 
  end subroutine get_random_coordinates

  !! Computes a Hamiltonian based on a single "s-like" orbitals per atom.
  ! @author Anders Niklasson
  ! @brief Computes a hamiltonian \f$ H_{ij} = (x/m)\exp(-(y/n + decay_{min}) |R_{ij}|^2))\f$, based on distances
  ! \f$ R_{ij} \f$. \f$ x,m,y,n,decay_{min} \f$ are fixed parameters.
  !
  ! @param coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
  ! @param types Index type for each atom in the system. Type for first atom = type[0] (not used yet)
  ! @return H 2D numpy array of Hamiltonian elements
  ! @param verb Verbosity. If True is passed, information is printed.
  !
  subroutine get_hamiltonian(coords,atomTypes,H,verb)
    implicit none 
    integer :: N,Nocc,m,hdim
    logical, intent(in) :: verb
    real(dp), allocatable :: xx(:)
    real(dp), allocatable, intent(in) :: coords(:,:)
    integer, allocatable, intent(in) :: atomTypes(:)
    real(dp), allocatable, intent(out) :: H(:,:) 
    real(dp) :: a,c,x,b,d,y,tmp,dist,eps,decay_min
    integer :: i,j,cnt

    hdim = size(coords,dim=2); Nocc = int(real(hdim)/4.0); eps = 1e-9; decay_min = 0.1; m = 78;
    a = 3.817632; c = 0.816371; x = 1.029769; n = 13;
    b = 1.927947; d = 3.386142; y = 2.135545;
    if(.not. allocated(H)) allocate(H(hdim,hdim))
    if(verb) write(*,*)"Constructing a simple Hamiltonian for the full system"
    cnt = 0
    do i = 1,hdim 
      x = mod((a*x+c),real(m))       
      y = mod((b*y+d),real(n))
      do j = 1,hdim 
        dist = norm2(coords(:,i)-coords(:,j))
        tmp = (x/m)*exp(-(y/n + decay_min)*(dist**2))
        H(i,j) = tmp
        H(j,i) = tmp
      enddo
    enddo
    return 
  end subroutine get_hamiltonian


end module proxy_a_mod


