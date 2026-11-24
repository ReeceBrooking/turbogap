! Dipole and polarisability readout for SOAP GAP models (forward pass only).
!
! STAGED FILE. Every routine below has its interface fixed, because the rest of
! the code already calls it. The bodies are yours: each one currently returns
! zeros so that the program compiles and runs. Replace everything between
! "YOUR CODE" and "END YOUR CODE"; declare any extra local variables you need
! under "Internal variables".
!
! Suggested order: 1 -> 6. After each one run
!     make && bin/tensor_test
! which checks the routines one at a time against hand-computed numbers.
!
! Maths and index formulas: docs/gap_tensor_readout_integration_guide.md, section 3.
! Python -> Fortran lookups: docs/python_to_fortran_cheatsheet.md.

module tensor_methods

  use types

  contains

!**************************************************************************
! 1. Sizes of the readout, derived from what the .gap file said.
!
!    Called once when the model is loaded and once per MPI rank, so that every
!    array extent in the code comes from this one place.
!
!    Set:  readout%n_dim      D   3 for a dipole, 6 for a polarisability
!          readout%n_cross    K1  cross_channels * n_species
!          readout%n_columns  K   dipole: C*T    polarisability: 2*C*T + K1*(K1+1)/2
!    From: readout%model_type (compare with model_dipole / model_polarisability),
!          readout%n_channels (C), readout%cross_channels (C_x), n_species (T)
  subroutine get_readout_sizes(readout, n_species)

    implicit none

!   Input variables
    integer, intent(in) :: n_species
!   In/out variables (inout, not out: the fields read from the file must survive)
    type(tensor_readout), intent(inout) :: readout
!   Internal variables

!   YOUR CODE
    select case( readout%model_type)
    case(model_dipole)
        readout%n_dim = 3
        readout%n_cross = 0
        readout%n_columns = readout%n_channels * n_species
    case(model_polarisability)
        readout%n_dim = 6
        readout%n_cross = readout%cross_channels * n_species
        readout%n_columns = 2 * readout%n_channels * n_species + readout%n_cross * (readout%n_cross + 1) / 2
    end select
!   END YOUR CODE

  end subroutine
!**************************************************************************




!**************************************************************************
! 2. Cosine cutoff: 1 at r = 0, falling smoothly to 0 at r = rcut.
!
!    A function, not a subroutine, because it returns one number and is used
!    inside expressions.
  function cosine_cutoff(r, rcut) result(fcut)

    implicit none

!   Input variables
    real*8, intent(in) :: r, rcut
!   Output variables
    real*8 :: fcut
!   Internal variables
    real*8 :: pi

!   YOUR CODE
    pi = dacos(-1.d0)
    fcut = 0.5d0*(1.d0 + dcos((pi * r) / rcut))
!   END YOUR CODE

  end function
!**************************************************************************




!**************************************************************************
! 3. Radial basis for ONE pair: warped Chebyshev polynomials times the cutoff.
!
!    radial(n) = T_{n-1}(x) * cosine_cutoff(r, rcut),   x = 2*(r/rcut - 1)**2 - 1
!
!    Called per neighbour (not for all neighbours at once): nothing is stored,
!    and the caller slices the first C_x channels for the cross group.
  subroutine radial_basis(r, rcut, n_channels, radial)

    implicit none

!   Input variables
    real*8, intent(in) :: r, rcut
    integer, intent(in) :: n_channels
!   Output variables
    real*8, intent(out) :: radial(1:n_channels)
!   Internal variables
    integer :: n
    real*8 :: x
!   YOUR CODE
    x = 2.d0 * ((r / rcut) - 1.d0)**2 - 1.d0
    radial(1) = 1.d0
    if (n_channels > 1) then
        radial(2) = x
    end if
    if (n_channels > 2) then
        do n = 3, n_channels
            radial(n) = 2.d0 * x * radial(n-1) - radial(n-2)
        end do
    end if
    radial = radial * cosine_cutoff(r, rcut)
!   END YOUR CODE

  end subroutine
!**************************************************************************




!**************************************************************************
! 4. Readout geometry v_i for every atom this MPI rank owns.
!
!    The neighbour arrays are the slices TurboGAP passes to every pair routine
!    (compare get_2b_energy_and_forces in gap.f90). Inside this routine they all
!    start at index 1:
!      n_neigh(i)           entries of atom i in the pair arrays, INCLUDING itself
!      rjs(k)               distance of pair k; the first entry of each atom is
!                           the atom itself, with rjs = 0 -> skip it
!      xyz(1:3, k)          r_j - r_i, used as stored (no sign flip)
!      neighbor_species(k)  species index t' of the neighbour, 1..n_species
!    The list is longer than the readout cutoff: keep only rjs(k) < readout%rcut.
!
!    v(1:D, 1:K, i) = [ bond basis | cross group ] of atom i. The site index is
!    last so that v(:, :, i) is one contiguous block.
!
!    This is the only routine that needs to know dipole from polarisability.
  subroutine get_readout_geometry(n_neigh, rjs, xyz, neighbor_species, n_species, readout, v)

    implicit none

!   Input variables
    real*8, intent(in) :: rjs(:), xyz(:, :)
    integer, intent(in) :: n_neigh(:), neighbor_species(:), n_species
    type(tensor_readout), intent(in) :: readout
!   Output variables
    real*8, intent(out) :: v(:, :, :)
!   Internal variables
    integer :: i, j, col, n, g, k, p, q, pair
    real*8 :: r_unit(1:3)
    real*8 :: radial(1:readout%n_channels)
    real*8 :: geom(1:6, 1:2)
    real*8 :: cross(1:3, readout%n_cross), a(1:3), b(1:3)
    real*8 :: tr
!   CROSS GROUP, declarations: you will need
!     - an array for this atom's cross vectors, 3 components x K1 slots
!       (K1 = readout%n_cross; it can be sized like radial is, from readout)
!     - integers for the two slots p, q and for a running column counter
!   Note: radial has C entries and the cross group reads the first C_x of
!   them, which is fine as long as C_x <= C.

!   YOUR CODE
    v = 0.d0
    select case( readout%model_type )
    case( model_dipole)
        k = 0
        do i = 1, size(n_neigh)
            do j = 1, n_neigh(i)
                k = k + 1
                if( j == 1 )cycle
                if( rjs(k) >= readout%rcut )cycle
                r_unit = xyz(:, k) / rjs(k)
                call radial_basis(rjs(k), readout%rcut, readout%n_channels, radial)
                do n = 1, readout%n_channels
                    col = (n - 1) * n_species + neighbor_species(k)
                    v(1:3, col, i) = v(1:3, col, i) + radial(n)*r_unit
                end do
            end do
        end do
    case(model_polarisability)
            k = 0
            do i = 1, size(n_neigh)
!               CROSS GROUP, step 1 (once per atom, before its neighbours):
!               set the cross-vector array to zero. It belongs to atom i only.
                cross = 0.d0
                do j = 1, n_neigh(i)
                    k = k + 1
                    if( j == 1 )cycle
                    if( rjs(k) >= readout%rcut )cycle
                    r_unit = xyz(:, k) / rjs(k)
                    geom(:, 1) = [1.d0, 1.d0, 1.d0, 0.d0, 0.d0, 0.d0]
                    geom(:, 2) = [r_unit(1)**2 - 1.d0/3.d0, r_unit(2)**2 - 1.d0/3.d0, r_unit(3)**2 - 1.d0/3.d0, &
                        r_unit(1) * r_unit(2), r_unit(2) * r_unit(3), r_unit(1) * r_unit(3)]
                    call radial_basis(rjs(k), readout%rcut, readout%n_channels, radial)
                    do n = 1, readout%cross_channels
                        col = (n - 1) * n_species + neighbor_species(k)
                        cross(1:3, col) = cross(1:3, col) + radial(n)*r_unit
                    end do
                    do g = 1, 2
                        do n = 1, readout%n_channels
                            col = (g - 1) * n_species * readout%n_channels + (n - 1) * n_species + neighbor_species(k)
                            v(1:6, col, i) = v(1:6, col, i) + radial(n)*geom(:, g)
                        end do
                    end do
!                   CROSS GROUP, step 2 (once per neighbour, still inside the j loop):
!                   add this pair to the cross vectors. Same as your dipole line,
!                   but into the cross-vector array, for n = 1 .. C_x only:
!                     slot = (n - 1)*n_species + neighbor_species(k)
!                     cross vector(slot) = cross vector(slot) + radial(n)*r_unit
                end do
!               CROSS GROUP, step 3 (once per atom, after ALL its neighbours):
!               the cross vectors are now complete sums. Turn them into columns:
!                 loop p = 1 .. K1, and inside it q = p .. K1   (pairs with p <= q)
!                 count the pairs as you go: 1, 2, 3, ...
!                 for each pair build the symmetric 3x3 product of vector p and
!                   vector q:  S(a,b) = 0.5*( vp(a)*vq(b) + vq(a)*vp(b) )
!                 subtract a third of its trace from the diagonal
!                 store (Sxx, Syy, Szz, Sxy, Syz, Szx) in column
!                   2*C*T + (pair count)   of v(1:6, :, i)
                pair = 0
                do p = 1, readout%n_cross
                    do q = p, readout%n_cross
                        pair = pair + 1
                        col = 2 * readout%n_channels * n_species + pair
                        a = cross(:, p)
                        b = cross(:, q)
                        tr = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
                        v(1:6, col, i) = [a(1)*b(1) - tr/3.d0, a(2)*b(2) - tr/3.d0, a(3)*b(3) - tr/3.d0, &
                            0.5d0*(a(1)*b(2) + a(2)*b(1)), 0.5d0*(a(2)*b(3) + a(3)*b(2)), 0.5d0*(a(3)*b(1) + a(1)*b(3))]
                    end do
                end do

            end do  
    end select
!   END YOUR CODE

  end subroutine
!**************************************************************************




!**************************************************************************
! 5. Site scalars of one SOAP block:  s_i(k) = delta**2 * sum_m (q_i . Q_m)**zeta * alpha(k, m)
!
!      soap(1:n_soap, 1:n_sites)        one COLUMN per atom, unit-normalised
!      Qs(1:n_soap, 1:n_sparse)         sparse points, one column each
!      alphas(1:n_cols, 1:n_sparse)     one column per sparse point
!      site_scalars(1:n_sites, 1:n_cols)
!
!    Two matrix products with an element-wise power in between;
!    local_property_predict (local_properties.f90) is the one-column version.
!    n_sites can be 0 (a chunk with no atom of this species): dgemm must not be
!    called then.
  subroutine get_site_scalars(soap, Qs, alphas, delta, zeta, site_scalars)

    implicit none

!   Input variables
    real*8, intent(in) :: soap(:, :), Qs(:, :), alphas(:, :), delta, zeta
!   Output variables
    real*8, intent(out) :: site_scalars(:, :)
!   Internal variables

!   YOUR CODE
    site_scalars = 0.d0
!   END YOUR CODE

  end subroutine
!**************************************************************************




!**************************************************************************
! 6. Contraction:  tensor(x) = tensor(x) + sum_i sum_k site_scalars(i, k) * v(x, k, i)
!
!      site_scalars(1:n_sites, 1:n_cols)
!      v(1:D, 1:n_cols, 1:n_sites)       the SAME atoms, in the same order
!      tensor(1:D)                       ADDED to: it is the running total over
!                                        all blocks and chunks (hence inout)
  subroutine contract_readout(site_scalars, v, tensor)

    implicit none

!   Input variables
    real*8, intent(in) :: site_scalars(:, :), v(:, :, :)
!   In/out variables
    real*8, intent(inout) :: tensor(:)
!   Internal variables

!   YOUR CODE
!   (nothing: adding zero leaves the running total unchanged)
!   END YOUR CODE

  end subroutine
!**************************************************************************

end module
