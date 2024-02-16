!> Library interface
!! \brief This file is used to interface to python via iso_c_binding 
!! library. 

!function proxya_get_hamiltonian(nats,norbs,coords_in,atomTypes_in,H_out,verb_in) result(err) bind(c, name='proxya_get_hamiltonian')
function proxya_get_hamiltonian(nats,norbs,coords_in,atomTypes_in,H_out,verb_in) result(err) bind(c, name='proxya_get_hamiltonian')
!function proxya_get_hamiltonian(nats) result(err) bind(c, name='proxya_get_hamiltonian')
    use iso_c_binding, only: c_char, c_double, c_int, c_bool
    use proxy_a_mod
    implicit none
    integer(c_int), intent(in), value  :: nats
    integer(c_int), intent(in), value  :: norbs
    real(c_double), intent(in)  :: coords_in(3*nats)
    integer(c_int), intent(in)  :: atomTypes_in(nats)
    logical(c_bool), intent(in), value :: verb_in
!    integer :: verb
    logical(c_bool) :: err
    real(c_double), intent(inout) :: H_out(norbs,norbs)

    real(dp), allocatable :: coords(:,:)
    integer, allocatable :: atomTypes(:)
    integer :: i
    real(dp), allocatable :: H(:,:)
    logical :: verb
   
    err = .true.

    allocate(coords(3,nats)) !indices will need to be flipped
    allocate(atomTypes(nats))
    allocate(H(norbs,norbs)) 
    
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

    call get_hamiltonian(coords,atomTypes,H,verb)

    H_out = H 

    err = .false.
    
end function proxya_get_hamiltonian


!isubroutine get_hamiltonian_bind(coords,atomTypes,H,verb)
!    use proxy_a_mod
!    implicit none
!    logical, intent(in) :: verb
!    real(dp), intent(in) :: coords(:,:)
!    integer, intent(in) :: atomTypes(:)
!    real(dp), intent(inout) :: H(:,:)
!
!    if (.not.ALLOCATED(CR)) ALLOCATE(CR(3,NATS))
!    CR = CR_IN
!
!    call get_hamiltonian(coords,atomTypes,H,verb)
!
!end subroutine get_hamiltonian_bind

