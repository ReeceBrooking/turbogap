# Python to Fortran: translations for the TNEP port

A lookup sheet for the NumPy and TensorFlow operations that appear most often
in `TNEP/gap_dipole.py`, `gap_models.py`, `data.py` and `TNEP.py`, and what to
write instead in TurboGAP's Fortran. Companion to
[gap_tensor_readout_integration_guide.md](gap_tensor_readout_integration_guide.md).

Fortran examples follow TurboGAP's habits: `real*8`, `d0` literals, lower
case.

---

## 1. The five differences that cause most bugs

| Topic | Python | Fortran |
|---|---|---|
| First index | 0 | 1 (by default) |
| Slice end | `a[i:j]` excludes `j` | `a(i:j)` **includes** `j` |
| Memory order | row-major: last index fastest | column-major: **first** index fastest |
| Number types | `1/3` is `0.333…` | `1/3` is `0` (integer division); `1.0/3.0` is single precision. Write `1.d0/3.d0` |
| Assignment | `b = a` is another name for the same data | `b = a` **copies** the array |

Two more that follow from these:

- A literal without `d0` is single precision. `0.5*x` silently loses digits;
  `0.5d0*x` does not. This alone can spoil a 1e-10 comparison.
- Variables are not zero until you set them. Python's `np.zeros` has no
  automatic equivalent.

## 2. Shapes and layout

A Python array `[N, D, K]` and a Fortran array `(D, K, N)` can hold the same
numbers. What matters is which index you loop over innermost.

| Situation | Rule |
|---|---|
| Choosing an index order | put the index you slice out whole **last** (`v(:, :, i)` is one contiguous block) |
| Loop nests | innermost loop over the **first** index |
| Reading a file Python wrote with `a.reshape(-1)` | Python `[N, D, K]` arrives in Fortran natural order as `(K, D, N)` |
| Python `[M, K]` written one row per line | read line `m` into column `m` of a `(K, M)` array |
| `a.reshape(...)` | Fortran `reshape` also exists but fills column-major, so the same call gives a **different** arrangement. Work out the index formula instead |
| `a.T` | `transpose(a)` for rank 2; for higher rank use `reshape(a, shape, order=[...])` or loops |

The flat index of element `(n, t)` in a Python `[C, T]` array is `n·T + t`.
The 1-based Fortran form is `(n − 1)·T + t`. Every column formula in the
integration guide is this pattern applied once or twice.

## 3. Creating and inspecting arrays

| Python | Fortran |
|---|---|
| `a = np.zeros((n, m))` | `allocate( a(1:n, 1:m) )` then `a = 0.d0` |
| `np.ones_like(a)` | allocate to the same shape, `= 1.d0` |
| `np.arange(n)` | `[(i, i = 1, n)]` (an implied-do array constructor) |
| `np.array([1., 2.])` | `[1.d0, 2.d0]` |
| `a.shape`, `tf.shape(a)` | `shape(a)`; one extent: `size(a, 1)` |
| `len(a)` | `size(a)` (total), `size(a, dim)` |
| `a.astype(np.float64)`, `tf.cast(x, tf.float64)` | `dfloat(i)` or `real(i, 8)` |
| `int(x)`, `round(x)` | `int(x)` truncates; `nint(x)` rounds |
| `np.pi` | `dacos(-1.d0)` (TurboGAP's habit) |
| `tf.constant(c)` | `real*8, parameter :: c = …` |
| `del a` | `deallocate(a)` |
| `a is None` | `allocated(a)` |

## 4. Element-wise maths

Whole-array expressions work as in NumPy when the shapes match exactly.

| Python | Fortran |
|---|---|
| `a + b`, `a * b`, `a / b` | same; `*` is element-wise, never a matrix product |
| `a ** 2`, `tf.square(a)` | `a**2` |
| `G ** xi` (integer `xi`) | `G**xi` with `xi` declared `integer`. A real exponent on a negative base gives NaN |
| `np.sqrt`, `np.exp`, `np.cos` | `dsqrt`, `dexp`, `dcos` (or the generic `sqrt`, `exp`, `cos`) |
| `np.abs` | `dabs` / `abs` |
| `np.minimum(a, b)`, `np.maximum` | `min(a, b)`, `max(a, b)` |
| `np.clip(x, lo, hi)` | `max(lo, min(hi, x))` |
| `np.where(mask, a, b)`, `tf.where` | `merge(a, b, mask)`, or a `where … elsewhere … end where` block |

## 5. Reductions

Axis numbers do not carry over: decide which Fortran index you mean.

| Python | Fortran |
|---|---|
| `a.sum()`, `tf.reduce_sum(a)` | `sum(a)` |
| `a.sum(axis=k)` | `sum(a, dim=k')` with `k'` the matching Fortran index, 1-based |
| `a.mean()` | `sum(a) / size(a)` |
| `a.max()`, `np.argmax(a)` | `maxval(a)`, `maxloc(a, 1)` |
| `np.linalg.norm(v)` | `norm2(v)` or `dsqrt(dot_product(v, v))` |
| `np.any(m)`, `np.all(m)` | `any(m)`, `all(m)` |
| `np.count_nonzero(m)` | `count(m)` |
| `np.allclose(a, b, atol=t)` | `maxval(dabs(a - b)) < t` |
| `np.trace(S)` | `S(1,1) + S(2,2) + S(3,3)` |

## 6. Linear algebra

| Python | Fortran |
|---|---|
| `a @ b`, `tf.matmul` (small) | `matmul(a, b)` |
| `a @ b` (large) | `dgemm`, as TurboGAP does ([local_properties.f90:209](../src/local_properties.f90#L209)) |
| `A.T @ B` | `dgemm("t", "n", …)`: the transpose is a flag, do not form `A.T` |
| `A @ x` (matrix times vector) | `matmul(A, x)` or `dgemv` |
| `np.dot(u, v)`, `einsum("i,i->", u, v)` | `dot_product(u, v)` |
| `np.outer(u, v)` | double loop, or `spread(u, 2, n) * spread(v, 1, m)` |
| `np.einsum(...)`, `tf.einsum` (anything else) | explicit loops. Write the index sum out; it is usually three lines |
| `np.linalg.solve`, `cholesky`, `inv` | LAPACK (`dposv`, `dpotrf`, `dgetri`). Fitting only; the forward pass needs none |

Reading a `dgemm` call:
`dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)` computes
`C = alpha · op(A) · op(B) + beta · C`, where `C` is `m × n` and `k` is the
shared inner dimension. `lda`, `ldb`, `ldc` are the **first extents of the
arrays as declared**, not of the transposed operands.

## 7. Indexing, gathering, scattering

This is where NumPy and TensorFlow code is most compact and the Fortran is
most different. The Fortran answer is almost always a plain loop.

| Python | Fortran |
|---|---|
| `a[idx]`, `tf.gather(a, idx)` | `a(idx)` with an integer array `idx` (a vector subscript). It makes a copy |
| `a[mask]` | `pack(a, mask)` for rank 1; otherwise loop and test |
| `D[ti == t]` (rows of one species) | loop over atoms, `if (species(i) == t)`, copy into the next free column of a work array |
| `np.flatnonzero(mask)` | `pack([(i, i = 1, n)], mask)` |
| `np.add.at(q, (i, t), R)`, `tf.math.unsorted_segment_sum` | loop over pairs: `q(:, t, i) = q(:, t, i) + R(:)` |
| per-centre segment sums (`starts`/`ends` loops) | the neighbour list is already grouped by centre: an outer loop over atoms with an inner loop over that atom's `n_neigh(i)` entries |
| `np.triu_indices(K)` | `do p = 1, K; do q = p, K` with a counter `c = c + 1` |
| `np.meshgrid` of neighbour pairs | two nested loops over one atom's neighbours |
| `a[:, None]`, `tf.newaxis`, broadcasting | **does not exist.** Loop, or `spread(a, dim, n)` |
| `np.concatenate([a, b])` (rank 1) | `[a, b]` |
| `np.concatenate([A, B], axis=…)` | allocate the result, assign the two slices |
| `np.stack([...])` | same: assign each piece into its slice |

On broadcasting: `R[:, :, None] * g[:, None, :]` in Python becomes a loop over
the two free indices in Fortran. Do not try to reproduce the reshapes that the
Python uses to set up a broadcast or a single `@`. They exist to avoid Python
loops, which is a problem Fortran does not have.

## 8. Control flow and program structure

| Python | Fortran |
|---|---|
| `for i in range(n)` | `do i = 1, n … end do` |
| `for k, x in enumerate(xs)` | `do k = 1, size(xs)` and index |
| `continue` / `break` | `cycle` / `exit` |
| `if a and b` | `if( a .and. b )then`. Both sides may be evaluated: no short-circuit guarantee |
| `==`, `!=` | `==`, `/=` |
| `def f(x): return y` | `subroutine f(x, y)` with `intent(in)` / `intent(out)`. Results come back through arguments |
| default argument | `optional` argument, tested with `present(arg)` |
| `cfg.field`, a dataclass | derived `type` with `%`: `cfg%field` |
| `raise ValueError(msg)` | `write(*,*) "ERROR: …"` then `stop` |
| `assert cond` | `if( .not. cond )` then the same |
| dict keyed by string | not available. Use named integer constants and a `select case` |
| `import module` | `use module` at the top, before `implicit none` |
| `@tf.function`, `tf.device`, `tf.Variable` | no equivalent needed: the code is compiled and the arrays are ordinary |

Declarations all go at the top of a routine, before the first executable
statement. Put `implicit none` in every routine: without it a misspelt
variable name is silently a new variable.

## 9. Traps specific to Fortran

- **Initialising in a declaration makes the variable permanent.**
  `integer :: k = 0` is set to 0 once, at program start, and keeps its value
  between calls. To reset on every call, assign in the body: `k = 0`.
- **Assumed-shape arguments restart at 1.** Pass `a(5:9)` to a dummy `x(:)`
  and inside the routine it is `x(1:5)`.
- **`intent(out)` allocatable arguments are deallocated on entry.** Use
  `intent(inout)` for something you accumulate into.
- **Unallocated arrays cannot be passed** to an ordinary dummy argument, or
  read. Check with `allocated(...)`.
- **Whole-array assignment to an allocatable reallocates it** to the new
  shape (`a = b`). Assignment to a section (`a(:) = b`) does not, and the
  shapes must already match.
- **Array sections with a stride or a vector subscript are copied** when
  passed to BLAS or MPI. It works, but gfortran warns about the temporary.
- **Integer overflow is silent.** Counts such as pairs × columns can exceed
  the default 32-bit integer.
- **Names are case-insensitive**, and `K` and `k` are the same variable. The
  maths uses both; pick different names (`n_cols`, `k`).

## 10. Files

| Python | Fortran |
|---|---|
| `np.savetxt(f, a)` | `open(newunit=u, file=f, status="unknown")`, then one `write(u, '(*(ES24.15))') a(:, i)` per line |
| `np.loadtxt(f)` | `open(…, status="old")`, then `read(u, *) a(:, i)` per line |
| end of file | `read(u, *, iostat=ios)`; `ios /= 0` means stop |
| counting lines | loop reading nothing until `iostat` is non-zero ([read_files.f90:513–520](../src/read_files.f90#L513)) |

`read(u, *)` is list-directed: it takes as many numbers as the variable list
needs and continues onto following lines if a line is too short. So a read of
one scalar from a line holding `K` numbers takes the first and discards the
rest, without an error.

Write with at least 17 significant digits (`ES24.15` gives 16) when the file
is a reference for a 1e-10 comparison.

## 11. House style observed in TurboGAP

The repository has no written style guide, formatter config or contributor
instructions. These are the habits of the existing code, worth matching in new
code:

| Topic | Convention |
|---|---|
| Reals | `real*8`, literals with `d0` |
| Intrinsics | the `d`-prefixed forms: `dsqrt`, `dabs`, `dcos`, `dacos`, `dfloat` |
| Case | lower case keywords and names, `snake_case` |
| Routines | subroutines inside a module after `contains`; `implicit none` in each one |
| Declarations | grouped under comments `!   Input variables`, `!   Output variables`, `!   Internal variables` |
| Errors | `write(*,*) "ERROR: …"` followed by `stop` |
| MPI | every MPI call inside `#ifdef _MPIF90 … #endif` |
| File header | the licence block starting `! HND XXXX`, copied at the top of each source file |
| Indentation | not uniform: 3 spaces in `types.f90`, `read_files.f90`, `gap_interface.f90`, `neighbors.f90`; 2 spaces in `turbogap.f90`, `gap.f90`, `mpi.f90`, `xyz.f90`. Match the file you are editing |
