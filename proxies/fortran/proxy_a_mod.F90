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
  public :: get_random_coordinates, get_densityMatrix

!!> Simple random number generator
!! This is important in order to compare across codes
!! written in different languages.
!!
!! To initialize:
!! \verbatim
!!   myRand = rand(123)
!! \endverbatim
!! where the argument of rand is the seed.
!!
!! To get a random number between "low" and "high":
!! \verbatim
!!   rnd = myRand.get_rand(low,high)
!! \endverbatim
!!
type, public :: rand
    integer :: a = 321
    integer :: b = 231
    integer :: c = 13
    integer :: seed
    integer :: stat
    contains 
    procedure :: init => rand_init
    procedure :: get_rand => rand_get_rand
end type rand
contains

  subroutine rand_init(self,seed)
    implicit none 
    class(rand), intent(inout) :: self
    integer, intent(in) :: seed
    self%seed = seed
    self%stat = seed*1000
  end subroutine rand_init

  function rand_get_rand(self,low,high) result(rnd)
    implicit none 
    class(rand), intent(inout) :: self
    real(dp) :: w
    real(dp) :: rnd
    real(dp), intent(in) :: low, high
    integer :: place 
    w = high - low
    place = self%a*self%stat
    place = int(place/self%b)
    rnd = real(mod(place,self%c))/real(self%c)
    place = rnd*1000000
    self%stat = place
    rnd = low + w*rnd
  end function rand_get_rand


  !!> Generating random coordinates 
  !! @brief Creates a system of size "nats = Number of atoms" with coordindates having 
  !! a random (-1,1) displacement from a simple cubic lattice with parameter 2.0 Ang.
  !!
  !! @param nats The total number of atoms
  !! @return coordinates Position for every atom. z-coordinate of atom 1 = coords[0,2]
  !!
  subroutine get_random_coordinates(nats,coords)
    implicit none 
    integer :: nats,ssize,length,atomsCounter
    integer :: i,j,k
    integer, allocatable :: seedin(:)
    real(dp), allocatable :: coords(:,:)
    real(dp) :: rnd,latticeParam
    type(rand) :: myrand
    !Get random coordinates
    length = int(real(nats)**(1.0/3.0)) + 1
    allocate(coords(3,nats))
    latticeParam = 2.0
    atomsCounter = 0
    call myrand%init(111)
    do i = 1,length
      do j = 1,length
        do k = 1,length
          atomsCounter = atomsCounter + 1
          if(atomsCounter > nats) exit
          rnd = myrand%get_rand(-1.0_dp,1.0_dp)
          coords(1,atomsCounter) = (i-1)*latticeParam + rnd
          write(*,*)coords(1,atomsCounter)
          rnd = myrand%get_rand(-1.0_dp,1.0_dp)
          coords(2,atomsCounter) = (j-1)*latticeParam + rnd
          rnd = myrand%get_rand(-1.0_dp,1.0_dp)
          coords(3,atomsCounter) = (k-1)*latticeParam + rnd
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
  subroutine get_hamiltonian(coords,atomTypes,H,S,verb)
    implicit none 
    integer :: n,Nocc,m,hdim
    logical, intent(in) :: verb
    real(dp), allocatable :: xx(:)
    real(dp), allocatable, intent(in) :: coords(:,:)
    integer, allocatable, intent(in) :: atomTypes(:)
    real(dp), allocatable, intent(out) :: H(:,:) 
    real(dp), allocatable, intent(out) :: S(:,:) 
    real(dp) :: a,c,x,b,d,y,tmp,dist,eps,decay_min
    integer :: i,j,k,cnt

    hdim = size(coords,dim=2); Nocc = int(real(hdim)/4.0); eps = 1e-9; decay_min = 0.1; m = 78;
    a = 3.817632; c = 0.816371; x = 1.029769; n = 13;
    b = 1.927947; d = 3.386142; y = 2.135545;
    if(.not. allocated(H)) allocate(H(hdim,hdim))
    if(verb) write(*,*)"Constructing a simple Hamiltonian for the full system"
    cnt = 0
    do i = 1,hdim 
      x = mod((a*x+c),real(m))       
      y = mod((b*y+d),real(n))
      do j = i,hdim
        dist = norm2(coords(:,i)-coords(:,j))
        tmp = (x/real(m))*exp(-(y/real(n) + decay_min)*dist**2)
        H(i,j) = tmp
        H(j,i) = tmp
      enddo
    enddo
    return 
  end subroutine get_hamiltonian

  !!> Computes the Density matrix from a given Hamiltonian.
  !! @author Anders Niklasson
  !! @brief This will create a "zero-temperature" Density matrix \f$ \rho \f$
  !! \f[ \rho  =  \sum^{nocc} v_k v_k^T \f]
  !! where \f$ v_k \f$ are the eigenvectors of the matrix \f$ H \f$
  !!
  !! @param H Hamiltonian mtrix 
  !! @param Nocc Number of occupied orbitals
  !! @param verb Verbosity. If True is passed, information is printed.
  !!
  !! @return D Density matrix
  !!
  subroutine get_densityMatrix(H,Nocc,D,verb)
#ifdef USEPROGRESS 
    use bml
    use prg_densitymatrix_mod
#endif
    
    implicit none 
    integer :: Nocc
    logical, intent(in) :: verb
    real(dp), allocatable, intent(in) :: H(:,:)
    real(dp), allocatable, intent(out) :: D(:,:)
    real(dp), allocatable :: Q(:,:), E(:),tmpmat(:,:),f(:,:)
    real(dp) :: mu
    integer :: info, lwork,i,j,k,norbs,homoIndex,lumoIndex 
    real(dp), allocatable :: work(:)
    real(dp) :: bndfil
    character(LEN=1), parameter :: jobz = 'V',  uplo = 'U'

#ifdef USEPROGRESS 
   type(bml_matrix_t) :: D_bml,H_bml
#endif 
    if(verb) write(*,*)"Computing the Density matrix"

    norbs = size(H,dim=1)
    
#ifdef USEPROGRESS
   bndfil = real(nocc)/real(norbs)
   call bml_zero_matrix(bml_matrix_dense,bml_element_real,&
          &dp,norbs,norbs,D_bml)
   call bml_zero_matrix(bml_matrix_dense,bml_element_real,&
          &dp,norbs,norbs,H_bml)
   call bml_import_from_dense(bml_matrix_dense,H,H_bml,0.0_dp,norbs)
   call prg_build_density_t0(H_bml,D_bml,0.0_dp,bndfil,E) 
   call bml_export_to_dense(D_bml,D)
   call bml_deallocate(D_bml)
   call bml_deallocate(H_bml)
#else
    lwork = 3*norbs - 1
    allocate(Q(norbs,norbs))
    allocate(work(lwork))
    allocate(E(norbs))
    Q = H
    call dsyev(jobz,uplo,norbs,Q,norbs,E,work,lwork,info)
    if(verb)write(*,*)"Eigenvalues",E
    homoIndex = Nocc
    lumoIndex = Nocc + 1
    mu = 0.5*(E(homoIndex) + E(lumoIndex))
    allocate(D(norbs,norbs))
    allocate(f(norbs,norbs))
    allocate(tmpmat(norbs,norbs))
    D = 0.0_dp
    f = 0.0_dp

    do i = 1,norbs
       if (E(i) < mu) then
         f(i,i) = 1.0_dp
       endif
    enddo

    CALL DGEMM('N', 'N', norbs, norbs, norbs, 1.0_dp, &
       tmpmat, norbs, Q, norbs, 0.0_dp, f, norbs) !Q*f
    CALL DGEMM('N', 'T', norbs, norbs, norbs, 1.0_dp, &
       tmpmat, norbs, Q, norbs, 0.0_dp, D, norbs) !(Q*f)*Qt

#endif

    return

  end subroutine get_densityMatrix


end module proxy_a_mod


