!> Library interface
!! \brief This file is used to interface to python via iso_c_binding 
!! library. 

!> Get Hamiltonian
!! \brief General fortran proxy function to get the Hamiltonian 
!! \param nats Number of atoms 
!! \param norbs Number of atoms 
!! \param coords_in Positions for all the atoms 
!! \param atomTypes Atom types indices for every atom in the system 
!! \param H_out Hamiltonian matrix 
!! \param S_out Overlap matrix 
!! \param verb_in Verbosity level 
!!
function proxya_get_hamiltonian(nats,norbs,coords_in,atomTypes_in,H_out,S_out,verb_in) &
                & result(err) bind(c, name='proxya_get_hamiltonian')
    use iso_c_binding, only: c_char, c_double, c_int, c_bool
    use proxy_a_mod
    implicit none
    integer(c_int), intent(in), value  :: nats
    integer(c_int), intent(in), value  :: norbs
    real(c_double), intent(in)  :: coords_in(3*nats)
    integer(c_int), intent(in)  :: atomTypes_in(nats)
    logical(c_bool), intent(in), value :: verb_in
    logical(c_bool) :: err
    real(c_double), intent(inout) :: H_out(norbs,norbs)
    real(c_double), intent(inout) :: S_out(norbs,norbs)

    real(dp), allocatable :: coords(:,:)
    integer, allocatable :: atomTypes(:)
    integer :: i
    real(dp), allocatable :: H(:,:)
    real(dp), allocatable :: S(:,:)
    logical :: verb
   
    err = .true.

    allocate(coords(3,nats)) !indices will need to be flipped
    allocate(atomTypes(nats))
    allocate(H(norbs,norbs)) 
    allocate(S(norbs,norbs)) 
    
    !Note that arrays appear in another order. We need to rearange 
    !the data. This is because of the column mayor (in python) vs. 
    !row mayor in fortran. 
    do i = 1, nats
      coords(1,i) = coords_in((i-1)*3 + 1)
      coords(2,i) = coords_in((i-1)*3 + 2)
      coords(3,i) = coords_in((i-1)*3 + 3)
    enddo

    atomTypes = atomTypes_in
   
    !A workaround to avoid fortran to c (one bit) boolean issues 
    if(verb_in .eqv. (1 == 1))then 
      verb = .true.
    else
      verb = .false.
    endif

    call get_hamiltonian(coords,atomTypes,H,S,verb)

    H_out = H 
    S_out = S

    err = .false.
    
end function proxya_get_hamiltonian


!> Get density matrix 
!! \brief General fortran proxy function to get the density matrix
!! \param norbs Number of atoms
!! \param nocc Number of occupied orbitals
!! \param ham_in Hamiltonian matrix input 
!! \param D_out Density matrix output 
!! \param verb_in Verbosity level
!!
function proxya_get_density_matrix(norbs,nocc,ham_in,D_out,verb_in) &
                & result(err) bind(c, name='proxya_get_density_matrix')
    use iso_c_binding, only: c_char, c_double, c_int, c_bool
    use proxy_a_mod
    implicit none
    integer(c_int), intent(in), value  :: norbs
    integer(c_int), intent(in), value  :: nocc
    real(c_double), intent(in)  :: ham_in(norbs*norbs)
    logical(c_bool), intent(in), value :: verb_in
    logical(c_bool) :: err
    real(c_double), intent(inout) :: D_out(norbs*norbs)

    real(dp), allocatable :: coords(:,:)
    integer, allocatable :: atomTypes(:)
    integer :: i
    real(dp), allocatable :: D(:,:)
    real(dp), allocatable :: ham(:,:)
    logical :: verb

    err = .true.

    allocate(D(norbs,norbs))
    allocate(ham(norbs,norbs))
    !From flatt to matrix 
    do i = 1,norbs
        ham(i,:) = ham_in(((i-1)*norbs + 1):i*norbs)
    enddo
        
    call get_densityMatrix(ham,Nocc,D,verb)
    !From matrix to faltt 
    do i = 1,norbs 
        D_out(((i-1)*norbs + 1):i*norbs) = D(i,:)
    enddo

end function proxya_get_density_matrix
