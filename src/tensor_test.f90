! STAGED: stand-alone checks for the routines in tensor_methods.f90.
!
!   make && bin/tensor_test
!
! Each check feeds a routine an input small enough to work out by hand and
! compares with the hand-computed answer. No model files or xyz files needed.
! The Makefile builds any src/<name>.f90 listed in PROGRAMS as a program linked
! against all modules, which is how this file gets access to tensor_methods.

program tensor_test

  use types
  use tensor_methods

  implicit none

  type(tensor_readout) :: readout
  real*8 :: radial(1:3), tensor(1:3), expected(1:6)
  real*8 :: soap(1:2, 1:1), Qs(1:2, 1:2), alphas(1:2, 1:2), site_scalars(1:1, 1:2), v_small(1:3, 1:2, 1:1)
  real*8, allocatable :: v(:, :, :)
!  One atom of species 1 with one neighbour of species 2 at distance 2 along +x.
!  The first entry of the list is the atom itself, as TurboGAP stores it.
  integer :: n_neigh(1:1) = [2], neighbor_species(1:2) = [1, 2]
  real*8 :: rjs(1:2) = [0.d0, 2.d0]
  real*8 :: xyz(1:3, 1:2) = reshape([0.d0, 0.d0, 0.d0, 2.d0, 0.d0, 0.d0], [3, 2])
  integer :: n_failed = 0

  write(*,*) "Checks for tensor_methods.f90 (r = 2, rcut = 4 throughout)"
  write(*,*)

! 1. Sizes ---------------------------------------------------------------------
  readout%model_type = model_dipole
  readout%n_channels = 10
  readout%cross_channels = 0
  call get_readout_sizes(readout, 2)
  call check_int("1. sizes, dipole C=10 T=2: D", readout%n_dim, 3)
  call check_int("1. sizes, dipole C=10 T=2: K", readout%n_columns, 20)
  readout%model_type = model_polarisability
  readout%cross_channels = 4
  call get_readout_sizes(readout, 2)
  call check_int("1. sizes, pol C=10 C_x=4 T=2: D", readout%n_dim, 6)
  call check_int("1. sizes, pol C=10 C_x=4 T=2: K1", readout%n_cross, 8)
  call check_int("1. sizes, pol C=10 C_x=4 T=2: K", readout%n_columns, 76)

! 2. Cutoff --------------------------------------------------------------------
  call check_real("2. cutoff at r = 0", [cosine_cutoff(0.d0, 4.d0)], [1.d0])
  call check_real("2. cutoff at r = rcut/2", [cosine_cutoff(2.d0, 4.d0)], [0.5d0])
  call check_real("2. cutoff at r = rcut", [cosine_cutoff(4.d0, 4.d0)], [0.d0])

! 3. Radial basis --------------------------------------------------------------
! x = 2*(0.5 - 1)**2 - 1 = -0.5, so T_0, T_1, T_2 = 1, -0.5, -0.5; cutoff = 0.5
  call radial_basis(2.d0, 4.d0, 3, radial)
  call check_real("3. radial basis, 3 channels", radial, [0.5d0, -0.25d0, -0.25d0])

! 4. Geometry, dipole ----------------------------------------------------------
! C = 2, T = 2, so K = 4. The neighbour is species 2: only columns 2 and 4 are
! filled, with R_1 = 0.5 and R_2 = -0.25 times the unit vector (1, 0, 0).
  readout%model_type = model_dipole
  readout%rcut = 4.d0
  readout%n_channels = 2
  readout%cross_channels = 0
  call get_readout_sizes(readout, 2)
  if( readout%n_dim == 3 .and. readout%n_columns == 4 )then
    allocate( v(1:3, 1:4, 1:1) )
    call get_readout_geometry(n_neigh, rjs, xyz, neighbor_species, 2, readout, v)
    call check_real("4. dipole geometry, column 1 (empty)", v(:, 1, 1), [0.d0, 0.d0, 0.d0])
    call check_real("4. dipole geometry, column 2", v(:, 2, 1), [0.5d0, 0.d0, 0.d0])
    call check_real("4. dipole geometry, column 3 (empty)", v(:, 3, 1), [0.d0, 0.d0, 0.d0])
    call check_real("4. dipole geometry, column 4", v(:, 4, 1), [-0.25d0, 0.d0, 0.d0])
    deallocate( v )
  else
    call skipped("4. dipole geometry (needs check 1 to pass first)")
  end if

! 4. Geometry, polarisability --------------------------------------------------
! C = 2, C_x = 1, T = 2: K = 8 bond columns + 3 cross columns = 11.
! Voigt order xx yy zz xy yz zx; the unit vector is (1, 0, 0).
  readout%model_type = model_polarisability
  readout%cross_channels = 1
  call get_readout_sizes(readout, 2)
  if( readout%n_dim == 6 .and. readout%n_columns == 11 )then
    allocate( v(1:6, 1:11, 1:1) )
    call get_readout_geometry(n_neigh, rjs, xyz, neighbor_species, 2, readout, v)
!   isotropic group: R * (1,1,1,0,0,0)
    expected = [1.d0, 1.d0, 1.d0, 0.d0, 0.d0, 0.d0]
    call check_real("4. pol geometry, isotropic n=1 (column 2)", v(:, 2, 1), 0.5d0*expected)
    call check_real("4. pol geometry, isotropic n=2 (column 4)", v(:, 4, 1), -0.25d0*expected)
!   traceless group: R * (u u^T - I/3)
    expected = [2.d0/3.d0, -1.d0/3.d0, -1.d0/3.d0, 0.d0, 0.d0, 0.d0]
    call check_real("4. pol geometry, traceless n=1 (column 6)", v(:, 6, 1), 0.5d0*expected)
    call check_real("4. pol geometry, traceless n=2 (column 8)", v(:, 8, 1), -0.25d0*expected)
!   cross group: slots are k1 = 1 (species 1, empty) and k1 = 2, v = (0.5, 0, 0).
!   Pairs (1,1), (1,2), (2,2) -> columns 9, 10, 11; only (2,2) is non-zero.
    call check_real("4. pol geometry, cross (1,1) (column 9)", v(:, 9, 1), 0.d0*expected)
    call check_real("4. pol geometry, cross (1,2) (column 10)", v(:, 10, 1), 0.d0*expected)
    call check_real("4. pol geometry, cross (2,2) (column 11)", v(:, 11, 1), 0.25d0*expected)
    call check_real("4. pol geometry, all species-1 bond columns empty", &
                    [maxval(dabs(v(:, 1:7:2, 1)))], [0.d0])
    deallocate( v )
  else
    call skipped("4. polarisability geometry (needs check 1 to pass first)")
  end if

! 5. Site scalars --------------------------------------------------------------
! One atom with descriptor (1, 0); sparse points (1, 0) and (0, 1), so the
! kernel row is (1, 0). delta = 2, zeta = 2: s = 4 * (first column of alphas).
  soap(:, 1) = [1.d0, 0.d0]
  Qs(:, 1) = [1.d0, 0.d0]
  Qs(:, 2) = [0.d0, 1.d0]
  alphas(:, 1) = [1.d0, 2.d0]
  alphas(:, 2) = [3.d0, 4.d0]
  call get_site_scalars(soap, Qs, alphas, 2.d0, 2.d0, site_scalars)
  call check_real("5. site scalars", site_scalars(1, :), [4.d0, 8.d0])

! 6. Contraction ---------------------------------------------------------------
! tensor = 4 * (1,0,0) + 8 * (0,1,0), added to a total that starts at (1,1,1).
  site_scalars(1, :) = [4.d0, 8.d0]
  v_small(:, 1, 1) = [1.d0, 0.d0, 0.d0]
  v_small(:, 2, 1) = [0.d0, 1.d0, 0.d0]
  tensor = 1.d0
  call contract_readout(site_scalars, v_small, tensor)
  call check_real("6. contraction (adds to the running total)", tensor, [5.d0, 9.d0, 1.d0])

  write(*,*)
  if( n_failed == 0 )then
    write(*,*) "All checks passed."
  else
    write(*,'(1X,I0,A)') n_failed, " check(s) not passing yet."
  end if

  contains

  subroutine check_int(label, got, want)
    character(len=*), intent(in) :: label
    integer, intent(in) :: got, want
    if( got == want )then
      write(*,'(1X,A,2X,A)') "PASS", label
    else
      write(*,'(1X,A,2X,A,A,I0,A,I0)') "FAIL", label, ": got ", got, ", expected ", want
      n_failed = n_failed + 1
    end if
  end subroutine

  subroutine check_real(label, got, want)
    character(len=*), intent(in) :: label
    real*8, intent(in) :: got(:), want(:)
    if( maxval(dabs(got - want)) < 1.d-12 )then
      write(*,'(1X,A,2X,A)') "PASS", label
    else
      write(*,'(1X,A,2X,A)') "FAIL", label
      write(*,'(9X,A,*(F10.6))') "got     ", got
      write(*,'(9X,A,*(F10.6))') "expected", want
      n_failed = n_failed + 1
    end if
  end subroutine

  subroutine skipped(label)
    character(len=*), intent(in) :: label
    write(*,'(1X,A,2X,A)') "SKIP", label
    n_failed = n_failed + 1
  end subroutine

end program
