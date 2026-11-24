! HND XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
! HND X
! HND X   TurboGAP
! HND X
! HND X   TurboGAP is copyright (c) 2019-2023, Miguel A. Caro and others
! HND X
! HND X   TurboGAP is published and distributed under the
! HND X      Academic Software License v1.0 (ASL)
! HND X
! HND X   This file, tensor_methods.f90, is copyright (c) 2026, <YOUR NAME HERE>
! HND X
! HND X   TurboGAP is distributed in the hope that it will be useful for non-commercial
! HND X   academic research, but WITHOUT ANY WARRANTY; without even the implied
! HND X   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! HND X   ASL for more details.
! HND X
! HND X   You should have received a copy of the ASL along with this program
! HND X   (e.g. in a LICENSE.md file); if not, you can write to the original
! HND X   licensor, Miguel Caro (mcaroba@gmail.com). The ASL is also published at
! HND X   http://github.com/gabor1/ASL
! HND X
! HND X   When using this software, please cite the following reference:
! HND X
! HND X   Miguel A. Caro. Phys. Rev. B 100, 024112 (2019)
! HND X
! HND XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

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
    readout%n_dim = 0
    readout%n_cross = 0
    readout%n_columns = 0
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

!   YOUR CODE
    fcut = 0.d0
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

!   YOUR CODE
    radial = 0.d0
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

!   YOUR CODE
    v = 0.d0
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
