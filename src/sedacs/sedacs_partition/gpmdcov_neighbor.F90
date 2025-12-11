module gpmdcov_neighbor_mod

  implicit none

  private

  integer, parameter, public :: dp = kind(1.0d0)
  integer, parameter, public :: low = kind(100)

  public ::  gpmdcov_get_vol, gpmdcov_get_nlist_box_indices

contains

  subroutine gpmdcov_get_vol(lattice_vectors,volBox)
    implicit none
    real(dp)                             ::  a1xa2(3), a2xa3(3), a3xa1(3)
    real(dp)                             ::  pi
    real(dp), intent(in)                 ::  lattice_vectors(:,:)
    real(dp), intent(inout)              ::  volBox

    volBox=0.0_dp

    pi = 3.14159265358979323846264338327950_dp

    a1xa2(1) = lattice_vectors(1,2)*lattice_vectors(2,3) - lattice_vectors(1,3)*lattice_vectors(2,2)
    a1xa2(2) = -lattice_vectors(1,1)*lattice_vectors(2,3) + lattice_vectors(1,3)*lattice_vectors(2,1)
    a1xa2(3) =  lattice_vectors(1,1)*lattice_vectors(2,2) - lattice_vectors(1,2)*lattice_vectors(2,1)

    a2xa3(1) = lattice_vectors(2,2)*lattice_vectors(3,3) - lattice_vectors(2,3)*lattice_vectors(3,2)
    a2xa3(2) = -lattice_vectors(2,1)*lattice_vectors(3,3) + lattice_vectors(2,3)*lattice_vectors(3,1)
    a2xa3(3) =  lattice_vectors(2,1)*lattice_vectors(3,2) - lattice_vectors(2,2)*lattice_vectors(3,1)

    a3xa1(1) = lattice_vectors(3,2)*lattice_vectors(1,3) - lattice_vectors(3,3)*lattice_vectors(1,2)
    a3xa1(2) = -lattice_vectors(3,1)*lattice_vectors(1,3) + lattice_vectors(3,3)*lattice_vectors(1,1)
    a3xa1(3) =  lattice_vectors(3,1)*lattice_vectors(1,2) - lattice_vectors(3,2)*lattice_vectors(1,1)

    !Get the volume of the cell
    volBox = lattice_vectors(1,1)*a2xa3(1)+ lattice_vectors(1,2)*a2xa3(2)+lattice_vectors(1,3)*a2xa3(3)

  end subroutine gpmdcov_get_vol  

  !! \brief It will bild a neighbor list using an "all to all" approach
  !! \param coords System coordinates. coords(1,7): x-coordinate of atom 7.
  !! \param lattice_vectors. Lattice vectors of the system box. lattice_vectors(1,3): z-coordinate of vector 1.
  !! \param nl Neighbor list type.
  !! \param verbose Verbosity level.
  !! \param rank MPI rank
  subroutine gpmdcov_get_nlist_box_indices(coords,outBoxOfI,lattice_vectors,nx,ny,nz,numParts,verbose,rank,numranks)
    implicit none
    integer                              ::  NBox, cnt, i, ibox
    integer                              ::  ith, ix, iy, iz
    integer                              ::  j, jbox, jj, jxBox
    integer                              ::  jyBox, jzBox, maxInBox, maxNeigh
    integer                              ::  myNumranks, myrank, nats, natsPerRank
    integer, intent(in)                  ::  nx, ny, nz
    integer, intent(inout) :: numParts
    integer :: tx
    integer                              ::  ty, tz,nx1,ny1,nz1,k
    integer, allocatable                 ::  inbox(:,:), ithFromXYZ(:,:,:)
    integer, allocatable, intent(out)     ::  outBoxOfI(:)
    integer, allocatable     ::  boxOfI(:),map(:)
    integer, allocatable                 ::  totPerBox(:), xBox(:), yBox(:), zBox(:)
    integer, intent(in)                  ::  verbose
    integer, optional, intent(in)        ::  numranks, rank
    real(dp)                             ::  coordsNeigh(3), density, distance, translation(3)
    real(dp)                             ::  volBox, minx, miny, minz, smallReal 
    real(dp)                             ::  maxx, maxy, maxz
    real(dp), allocatable, intent(in)    ::  coords(:,:), lattice_vectors(:,:)
    real(dp)                 ::  rcutx,rcuty,rcutz
    logical, allocatable :: boxLog(:)
#ifdef DO_MPI
    integer, allocatable :: rankRange(:,:)
#endif

    if(present(rank).and.present(numranks))then
      myrank = rank
      myNumranks = numranks
    else
      myrank = 1
      myNumranks = 1
    endif

    nats = size(coords,dim=2) !Get the number of atoms
    natsPerRank = int(nats/myNumranks)

    !We will have approximatly [(4/3)*pi * rcut^3 * atomic density] number of neighbors.
    !A very large atomic density could be 1 atom per (1.0 Ang)^3 = 1 atoms per Ang^3  
    call gpmdcov_get_vol(lattice_vectors,volBox)
    density = 1.0_dp
    maxneigh = int(floor(3.14592_dp * (4.0_dp/3.0_dp) * density * (rcutx*rcuty*rcutz)))

    minx = 1.0d10
    miny = 1.0d10
    minz = 1.0d10
    maxx = -1.0d10
    maxy = -1.0d10
    maxz = -1.0d10
    do i = 1,nats
       minx = min(minx,coords(1,i))
       miny = min(miny,coords(2,i))
       minz = min(minz,coords(3,i))
       maxx= max(maxx,coords(1,i))
       maxy = max(maxy,coords(2,i))
       maxz = max(maxz,coords(3,i))
    enddo

    !We assume the box is orthogonal
    rcutx = (maxx - minx)/(real(nx)) 
    rcuty = (maxy - miny)/(real(ny)) 
    rcutz = (maxz - minz)/(real(nz)) 

    NBox = nx*ny*nz
    maxInBox = int(density*(rcutx*rcuty*rcutz)) !Upper boud for the max number of atoms per box
    allocate(inbox(NBox,maxInBox))
    inbox = 0
    allocate(totPerBox(Nbox))
    totPerBox = 0
    allocate(boxOfI(nats))
    boxOfI = 0
    allocate(xBox(Nbox))
    xBox = 0
    allocate(yBox(Nbox))
    yBox = 0
    allocate(zBox(Nbox))
    zBox = 0
    allocate(ithFromXYZ(nx,ny,nz))
    ithFromXYZ = 0

    smallReal = -0.0000001_dp
    !Search for the box coordinate and index of every atom
    do i = 1,nats
      !Index every atom respect to the discretized position on the simulation box.
      !tranlation = coords(:,i) - origin !For the general case we need to make sure coords ar > 0 
      ix = 1 + int(floor(abs((coords(1,i) - minx + smallReal)/(rcutx)))) !small box x-index of atom i
      iy = 1 + int(floor(abs((coords(2,i) - miny + smallReal)/(rcuty)))) !small box y-index //
      iz = 1 + int(floor(abs((coords(3,i) - minz + smallReal)/(rcutz)))) !small box z-index //

!      write(*,*)coords(:,i),ix,iy,iz,(coords(3,i) - minz + smallReal)/rcutz
      if(ix > nx .or. ix < 0)then 
              write(*,*)"ix",ix
              Stop "Error in box index"
      endif
      if(iy > ny .or. iy < 0)then 
              write(*,*)"iy",iy
              Stop "Error in box index"
      endif
      if(iz > nz .or. iz < 0)then 
              write(*,*)"iz",iz
              Stop "Error in box index"
      endif

      ith =  ix + (iy-1)*(nx) + (iz-1)*(nx)*(ny)  !Get small box index
      boxOfI(i) = ith

      !From index to box coordinates
      xBox(ith) = ix
      yBox(ith) = iy
      zBox(ith) = iz

      !From box coordinates to index
      ithFromXYZ(ix,iy,iz) = ith

      totPerBox(ith) = totPerBox(ith) + 1 !How many per box
      if(totPerBox(ith) > maxInBox) Stop "Exceeding the max in box allowed"
      inbox(ith,totPerBox(ith)) = i !Who is in box ith
!      write(*,*)ix,iy,iz,boxOfI(i),coords(:,i)

    enddo

    !Do a remaping
    allocate(boxLog(nx*ny*nz))
    boxLog = .false. 
    do k = 1,nats
        boxLog(boxOfI(k)) = .true.
    enddo
    numParts = 0
    allocate(map(nx*ny*nz))
    do k = 1,nx*ny*nz
        if(boxLog(k))then 
                numParts = numParts + 1
                map(k) = numParts !From old numeration k to new numeration map(k)
        endif
    enddo
    deallocate(boxLog)

    !Recoloring
    allocate(outBoxOfI(nats))
    do k = 1,nats
        outBoxOfI(k) = map(boxOfI(k))
    enddo
    deallocate(map)

    end subroutine gpmdcov_get_nlist_box_indices

end module gpmdcov_neighbor_mod
