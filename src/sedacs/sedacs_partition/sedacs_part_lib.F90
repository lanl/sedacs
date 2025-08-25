function sedacs_part_fortran(nnodes_in,maxDegs_in,whichParts_guess_inout,graph_in,degs_in&
&,nparts_in,verbose_in) result(err) bind(c, name='sedacs_part')

  use iso_c_binding, only: c_char, c_double, c_int, c_bool
  use prg_graph_mod
    
  implicit none
  integer(c_int), intent(in), value  :: nnodes_in, maxDegs_in, nparts_in, verbose_in 
  integer(c_int), intent(inout)  :: whichParts_guess_inout(nnodes_in) 
  integer(c_int), intent(inout)  :: degs_in(nnodes_in) 
  integer(c_int), intent(inout)  :: graph_in(nnodes_in*maxDegs_in)
  logical(c_bool) :: err

  integer :: k, l
  integer, allocatable :: whichParts_guess(:), degs(:)
  integer, allocatable :: graph(:,:)

  allocate(whichParts_guess(nnodes_in))
  allocate(degs(nnodes_in))
  allocate(graph(nnodes_in,maxDegs_in)) 

  whichParts_guess(:) = whichParts_guess_inout(:)

  degs(:) = degs_in(:)

  do l = 1, nnodes_in 
    do k = 1, maxDegs_in
      graph(l, k) = graph_in((l-1)*maxDegs_in + k)
    enddo
  enddo

!  do k = 1, nnodes_in 
!    do l = 1, maxDegs_in
!      graph(l, k) = graph_in((k-1)*maxDegs_in + l)
!    enddo
!  enddo

  call prg_sedacsPartition(whichParts_guess,graph,degs,nparts_in,nnodes_in,verbose_in)

  whichParts_guess_inout(:) = whichParts_guess(:)

  if (allocated(whichParts_guess)) deallocate(whichParts_guess)
  if (allocated(degs)) deallocate(degs)
  if (allocated(graph)) deallocate(graph)

  return


end function sedacs_part_fortran


function sedacs_nlistbox_fortran(nats_in,coords_in,boxOfI_out,latticeVectors_in&
&,nx_in,ny_in,nz_in,numparts_out,verbose_in,rank_in,numranks_in) result(err) bind(c, name='sedacs_nlistbox')

  use iso_c_binding, only: c_char, c_double, c_int, c_bool
  use gpmdcov_neighbor_mod

  implicit none
  integer(c_int), intent(in), value  :: nats_in, nx_in, ny_in, nz_in, rank_in, numranks_in
  integer(c_int), intent(inout)  :: numparts_out
  real(c_double), intent(inout)  :: coords_in(3*nats_in)
  integer(c_int), intent(inout)  :: boxOfI_out(nats_in)
  real(c_double) ,intent(inout) :: latticeVectors_in(9)
  integer(c_int), intent(in), value :: verbose_in
  logical(c_bool) :: err

  logical :: err_status, is_nan  
  integer :: k
  integer, allocatable  :: boxOfI(:)
  real(dp), allocatable :: coords(:,:)
  real(dp), allocatable :: latticeVectors(:,:)

  err = .true.
  allocate(boxOfI(nats_in))
  allocate(coords(3,nats_in))
  allocate(latticeVectors(3,3))

  !Note that arrays appear in another order. We need to rearange
  !the data. This is because of the column mayor (in python) vs.
  !row mayor in fortran.
  do k = 1, nats_in
    coords(1,k) = coords_in((k-1)*3 + 1)
    coords(2,k) = coords_in((k-1)*3 + 2)
    coords(3,k) = coords_in((k-1)*3 + 3)
  enddo

  latticeVectors(1,1) = latticeVectors_in(1)
  latticeVectors(1,2) = latticeVectors_in(2)
  latticeVectors(1,3) = latticeVectors_in(3)

  latticeVectors(2,1) = latticeVectors_in(4)
  latticeVectors(2,2) = latticeVectors_in(5)
  latticeVectors(2,3) = latticeVectors_in(6)

  latticeVectors(3,1) = latticeVectors_in(7)
  latticeVectors(3,2) = latticeVectors_in(8)
  latticeVectors(3,3) = latticeVectors_in(9)

  call gpmdcov_get_nlist_box_indices(coords,boxOfI,latticeVectors,nx_in,ny_in,nz_in,numparts_out,verbose_in,rank_in,numranks_in)

  boxOfI_out(:) = boxOfI(:)

  deallocate(boxOfI)
  deallocate(coords)
  deallocate(latticeVectors)

  err = err_status

  return

end function sedacs_nlistbox_fortran 
