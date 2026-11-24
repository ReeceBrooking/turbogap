# Dipole and polarisability prediction — where everything slots in

Companion to [gap_tensor_readout_plan.md](gap_tensor_readout_plan.md). The plan
says *what* to build and in which order. This file says *where* each piece
attaches to TurboGAP, what arrays you are handed at that point, and what you
must hand back. It contains no routine bodies.

Line numbers refer to commit `93eb9d6` (branch `brookite`). They will drift as
you edit; the routine names and the quoted landmarks will not.

The maths is in [gap_forward_pass.md](gap_forward_pass.md), called "the
guide" below, with its section numbers. Section 3 here only adds the 1-based
index forms and the Python line references. The Python itself is
`~/unix/TNEP/gap_dipole.py` and the geometry helpers in `data.py`
(lines 395–553).

## Status: the plumbing is staged

Everything in sections 5 to 7 is already in the code, marked with comments
starting `STAGED`. What is left is the six routine bodies in
[src/tensor_methods.f90](../src/tensor_methods.f90), in the order given
there. Check them with

```
make && bin/tensor_test
```

which tests each routine against hand-computed numbers and needs no model
files. Sections 5 to 7 now serve as the explanation of what the staged code
does and why. Two details differ from the text below, both simplifications:

- `get_gap_soap` receives the whole `v_readout` array as an `allocatable`
  dummy, which keeps the caller's bounds. Its last index is the global atom
  id, so the staged code indexes it with `in_to_out_site(i)` directly and the
  `− this_i_beg + 1` offset of 5.3 is not needed.
- `write_extxyz` is called once, with the tensor as an optional argument that
  is empty for an energy model.

## Scope of this version

SOAP site models only. No 2b or 3b site models, no triplet loop, no angular
kernel.

| | dipole | polarisability |
|---|---|---|
| readout columns | bond basis, one group | isotropic group, traceless group, cross group |
| `D` | 3 | 6 (Voigt `xx yy zz xy yz zx`) |
| `K` | `C·T` | `2·C·T + K1 (K1 + 1) / 2`, `K1 = C_x·T` |

Hard-coded for now:

- bond vector: unit vector `û = d/|d|` everywhere;
- cutoff function: cosine, `f_c = (1 + cos(π r / r_c)) / 2`;
- radial family: warped Chebyshev (guide 3.2);
- species-pair columns: on.

Read from the model file at startup: target, `r_c`, `C`, `C_x`, and per block
ξ, δ, `K` and the two data files.

A second SOAP scale (the 6 Å blocks of the polarisability model) needs no
extra code: it is two more `soap_turbo` blocks reading out through the same
`v_i`.

---

## 0. Read this first

1. **The models must be fitted with these settings.** The production dipole
   model in guide section 1 uses full `d`, cosine², a 2b site model and a
   dipole cross group. None of that is in this scope. A dipole model for this
   port needs `dipole_unit_vector = True`, `dipole_cutoff = "cosine"`,
   `gap_site_models = ["soap"]`, `gap_cross_channels = 0`, and the bond-vector
   sign flipped to TurboGAP's convention (2.1). The production
   polarisability model needs refitting without its 3b block. Have the reader
   stop with an error if the model file says anything else (5.1), so a
   mismatched model cannot run quietly.
2. **The exporter the guide describes is not in the TNEP checkout.** Guide
   section 7.7 names `model_io.save_gap_model` / `load_gap_model`. Neither
   exists in any `.py` file under `~/unix` (branch
   `ExperimentalPolarisability`, commit `9030017`), and `GapDipoleModel` is
   in-memory only. Phase 0 is real work. Section 8 says which Python objects
   to write.
3. **The tensor SOAP blocks use the standard keywords `desc_sparse` and
   `alphas_sparse`** (guide 7.7), the same ones an energy GAP uses. With
   `do_prediction = .true.` the existing reader would open the `K`-column α
   file, read the first number of each line, and compute "energies" from it
   without any error. Section 5.2 shows where to branch.
4. **One sparse set and one α block per species.** So one `soap_turbo` block
   per central species, as TurboGAP already expects.
5. **`get_gap_soap` does not see all of a rank's atoms at once.** The main loop
   splits each rank's sites into memory-limited chunks, and inside the routine
   the sites are filtered by central species. Section 5.3 gives the index
   mapping you need to pick the right `v_i`.
6. **The 2D α reader cannot be a new branch of
   `read_alphas_and_descriptors`.** Its `alphas` dummy is rank 1
   (`read_files.f90:506`) and four existing calls pass rank-1 arrays. Write a
   sibling routine instead (section 4.3).
7. **`tensor_methods.f90` is not in the build yet.** It is an empty file and is
   missing from `SRC` in the [Makefile](../Makefile).
8. **The forward pass is the small part.** It is one neighbour loop and two
   matrix products. Most of the work, and most of the ways to get a silent
   wrong answer, are in how the model parameters travel from the `.gap` file
   to the place they are used. Section 5.0 follows every parameter through
   that path; read it before the per-file sections.

---

## 1. The main program, as a map

Everything happens in [src/turbogap.f90](../src/turbogap.f90), one long
program. These are the landmarks you will touch, in execution order.

| Lines | What happens | Your change |
|---|---|---|
| [412](../src/turbogap.f90#L412) | `read_input_file` fills `params`. Runs on **every** rank. | none: the model type comes from the `.gap` file, not the input file |
| [466](../src/turbogap.f90#L466) | `read_gap_hypers` reads the `.gap` file. Runs on **rank 0 only**. | one new argument for the readout config |
| [523–805](../src/turbogap.f90#L523) | Broadcast of everything `read_gap_hypers` produced | broadcast your new fields (section 6) |
| after [805](../src/turbogap.f90#L805) | End of the broadcast section | for a tensor model, turn the energy path off on every rank (5.4) |
| [942](../src/turbogap.f90#L942) | Start of the frame loop (`do while (repeat_xyz …)`) | none |
| [1243 / 1262](../src/turbogap.f90#L1243) | `build_neighbors_list` (MPI and serial variants) | none |
| [1305](../src/turbogap.f90#L1305) | `if (do_prediction .or. write_soap .or. write_derivatives)` — the block that holds all the GAP work | add "is a tensor model" to this condition |
| [1360–1367](../src/turbogap.f90#L1360) | Per-frame zeroing of the energy accumulators | zero your tensor accumulator here |
| [1521–1523](../src/turbogap.f90#L1521) | Just before the SOAP loop | **geometry call goes here** |
| [1525–1794](../src/turbogap.f90#L1525) | Loop over `soap_turbo` blocks; inner loop over chunks; `get_gap_soap` call at [1586](../src/turbogap.f90#L1586) | pass geometry, model and accumulator into `get_gap_soap` |
| after [1794](../src/turbogap.f90#L1794) | End of the SOAP loop | `mpi_reduce` your 3 or 6 numbers |
| [3394–3424](../src/turbogap.f90#L3394) | Static-prediction output: `get_xyz_energy_string` then `write_extxyz` | extend the condition, pass the tensor |
| [3662, 3684](../src/turbogap.f90#L3662) | MD output calls of `write_extxyz` | only if you want μ(t) along a trajectory later |

The frame loop already handles multi-frame xyz files. One pass through the
loop is one frame, so "per frame" in the plan means "once per pass, after
line 1297".

---

## 2. What TurboGAP hands you

### 2.1 Neighbour arrays

Produced by `build_neighbors_list`
([neighbors.f90:207](../src/neighbors.f90#L207)). All of these are flat arrays
over atom pairs, ordered by centre atom.

| Array | Shape on this rank | Meaning |
|---|---|---|
| `n_neigh` | `(1:n_sites)`, global, valid on all ranks | number of list entries of atom `i`, **including itself** |
| `rjs` | `(j_beg:j_end)` | distance of each pair |
| `xyz` | `(1:3, j_beg:j_end)` | **`r_j − r_i`** ([neighbors.f90:57](../src/neighbors.f90#L57): `dist = posj - posi`) |
| `neighbor_species` | `(j_beg:j_end)` | species index of the neighbour, `1..n_species` |
| `neighbors_list` | `(j_beg:j_end)` | atom id of the neighbour; may exceed `n_sites` when a supercell was built |
| `i_beg, i_end` | scalars | this rank's atoms |
| `j_beg, j_end` | scalars | this rank's pairs; `j_beg = 1` always, the pair arrays are rank-local |

How to walk it. This idiom is used everywhere, for example
[gap.f90:260–270](../src/gap.f90#L260):

- keep a running pair counter `k`, starting at 0;
- for each site `i`, the first entry (`k + 1`) is the atom itself with
  `rjs = 0`; skip it;
- entries `2 .. n_neigh(i)` are the real neighbours.

Inside a subroutine that received `n_neigh(i_beg:i_end)` and
`rjs(j_beg:j_end)` as assumed-shape dummies, both start at index 1 again. That
is why the existing routines never see `i_beg`.

Four things to do on entry:

1. **Use `xyz` as it is** and divide by `rjs(k)` for `û`. No sign flip: the
   fit is being changed to TurboGAP's convention, `d_ij = r_j − r_i` (see the
   note below). This replaces the "negate on entry" row in the plan's
   section 5.
2. **Skip** the self entry. It has `rjs = 0`, so it would also divide by zero.
3. **Filter** `rjs(k) < r_c`. The list is built at `rcut_max` plus buffer,
   which is at least as large as every block cutoff.
4. **Species** come from `neighbor_species(k)` directly. Do not index a
   per-atom array with `neighbors_list(k)`: with a supercell that id is out of
   range.

**The sign convention is settled on the Python side.** TurboGAP stores the
vector from the centre to the neighbour (`r_j − r_i`). The Python currently
uses the opposite (`d = -D` at `data.py:483`, and again in
`gap_models._site_pairs`). The decision is to flip the Python and refit, so
the Fortran needs no sign handling at all.

What that means in practice:

- A dipole model fitted **before** the flip gives exactly `−μ` in TurboGAP,
  because every dipole column is linear in `û`. Only export dipole models
  fitted after it.
- Polarisability models are unaffected either way: `û` only ever appears
  twice in a product (`û ûᵀ`, and `v^p v^qᵀ` in the cross group), so the sign
  cancels. Old and new fits give the same χ.
- The reference dumps for the dipole (section 8) must come from the flipped
  Python, or `V_i` will disagree by a sign in every column.

Edge case, only for cells shorter than `r_c`: the Python drops pairs where the
neighbour is a periodic image of the centre itself (`keep = i != j`,
`data.py:482`). TurboGAP keeps them as ordinary neighbours. They cancel for
the dipole but not for the polarisability. To match exactly, also skip entries
where `mod(neighbors_list(k) - 1, n_sites) + 1` equals the centre atom. For
your water boxes it never triggers.

### 2.2 Species indices

The species index of an atom is the position of its element in the
`species = …` line of the `input` file
([read_files.f90:202](../src/read_files.f90#L202)). The Python uses
`types = [8, 1]`, so index 0 is O and index 1 is H. The input file must say
`species = O H`, and each `soap_turbo` block must list its species in the same
order, because `central_species` indexes the block's own list.

### 2.3 The SOAP descriptor

Inside `get_gap_soap`, after the `get_soap` call
([gap_interface.f90:241](../src/gap_interface.f90#L241)):

| Name | Shape | Meaning |
|---|---|---|
| `soap` | `(1:n_soap, 1:n_sites)` | one **column** per kept site, unit-normalised, compressed |
| `n_sites` | scalar | number of sites in this chunk that have this block's central species |
| `in_to_out_site(i)` | `(1:n_all_sites)` | for `i ≤ n_sites`: the **global** atom id of kept site `i` |

The Python `D` is `[N, Q]` with one **row** per atom. Transposed layout, same
numbers.

---

## 3. The contracts, in Fortran indexing

The formulas are in the guide, sections 3 and 4, with 0-based indices. Below
is each one in scope again, with the Python line it comes from and the 1-based
column index. Where they disagree, the guide and the Python win.

| Here | Guide |
|---|---|
| 3.1 radial | 3.2 |
| 3.2 bond basis | 3.3, 3.4 |
| 3.3 cross group | 3.5 (polarisability form) |
| 3.4 kernel and contraction | 3.9 |

`T` = number of species, `C` = radial channels, `C_x` = cross channels. All
indices below are 1-based.

### 3.1 Radial weights

`_radial_weights`, `data.py:446`. For each pair, `n = 1..C`:

```
x      = 2 (r/r_c − 1)² − 1
R_n(r) = T_{n−1}(x) · f_c(r),      f_c = (1 + cos(π r / r_c)) / 2
```

Chebyshev by recurrence (guide 3.2). The channels are nested: the first `C_x`
channels are the cross-group basis. So evaluate once per pair at
`max(C, C_x)` and slice.

### 3.2 Bond basis `V_i`

`bond_geometry` (`data.py:428`), `_neighbor_dipole_V` (`data.py:535`),
`_collapse_V` (`gap_dipole.py:162`).

Per pair, the geometric factor `g(û)`:

| Target | Groups `G` | `g` |
|---|---|---|
| dipole | 1 | `û` |
| polarisability | 2 | group 1 (isotropic): `(1, 1, 1, 0, 0, 0)`; group 2 (traceless): `(û_x², û_y², û_z², û_x û_y, û_y û_z, û_z û_x) −` group 1 `/ 3` |

Sum over neighbours, into the column

```
k = ((g − 1)·C + (n − 1))·T + t'        width  K_bond = G·C·T
```

`t'` is the neighbour's species and varies fastest.

### 3.3 Cross group (polarisability only)

`_cross_vectors` (`gap_dipole.py:172`), `_cross_group` (line 184),
`_finish_v` (line 223). Appended after the bond columns.

First the cross vectors, one 3-vector per slot:

```
v^{k1} = Σ_{j of species t'} R_a(r_ij) · û_ij        k1 = (a − 1)·T + t',   K1 = C_x·T
```

For `C_x ≤ C` these are exactly the first `K1` columns of the **dipole** bond
basis. One accumulation routine serves both.

Then one Voigt column per pair `p ≤ q`, in row-major upper-triangle order
(`(1,1), (1,2), …, (1,K1), (2,2), …`):

```
S = ½ (v^p v^qᵀ + v^q v^pᵀ),    S ← S − tr(S)/3 · I,    column = (S_xx, S_yy, S_zz, S_xy, S_yz, S_zx)
```

Width `K1 (K1 + 1) / 2`. Then `v_i = [ V_i | cross ]`.

Note this is a product of two per-atom sums, built **after** the neighbour
loop for that atom has finished. It is not a sum over pairs.

### 3.4 Kernel and contraction

`_phi_frame`, `gap_dipole.py:314`, written there as a design matrix. The
per-atom form (guide 3.9):

```
s_i(k) = δ² · Σ_m (q_i · Q_m)^ξ · α(m, k)
y(x)  += Σ_k s_i(k) · v_i(x, k)
```

Every block, including a second SOAP scale, contracts with the same `v_i`. No
offset, no clamp. δ and ξ arrive as the block's `delta` and `zeta` keywords.
ξ = 2 in production.

---

## 4. `tensor_methods.f90`: suggested routines

One module, `use types` at most. Names are suggestions.

The site-indexed array uses the site as the **last** index. Fortran is
column-major, so `v(:, :, i)` is then a contiguous `D × K` block you can hand
straight to BLAS.

### 4.1 Geometry (phase 1)

| Routine | In | Out |
|---|---|---|
| radial basis | `r` (scalar), `r_c`, number of channels | `R(1:n_channels)`, already multiplied by `f_c` |
| readout geometry | `n_neigh(:)`, `rjs(:)`, `xyz(:,:)`, `neighbor_species(:)`, readout config | `v(1:D, 1:K, 1:n_sites_local)` |

Notes:

- All four neighbour arrays are `intent(in)` assumed-shape. Copy the argument
  style of `get_2b_energy_and_forces`
  ([gap.f90:210–221](../src/gap.f90#L210)): it takes this set.
- Make `v` an `intent(out)` array allocated by the caller. Then the routine
  never needs to know about `i_beg`.
- This routine is the **only** place that branches on dipole versus
  polarisability (`select case` on the model type of 5.0). Everything outside
  it works with `D` and `K` and does not care which model it is.
- For the polarisability, a small local `(3, K1)` work array per atom holds
  the cross vectors during the neighbour loop; the cross columns are filled
  from it once the loop over that atom's neighbours ends.

### 4.2 Kernel and contraction (phase 3)

| Routine | In | Out |
|---|---|---|
| site scalars | `q(1:dim, 1:n)`, `Qs(1:dim, 1:M)`, `alphas(1:K, 1:M)`, `delta`, `zeta` | `s(1:n, 1:K)` |
| contraction | `s(1:n, 1:K)`, geometry for those `n` sites | `y(1:D)`, `intent(inout)`, accumulated |

`dgemm` hints for the site scalars:

- First product: identical to
  [local_properties.f90:209](../src/local_properties.f90#L209). `"t", "n"`
  gives `Kmat(n, M) = qᵀ Qs`.
- Power: `local_property_predict`
  ([line 213](../src/local_properties.f90#L213)) shows the integer-exponent
  path, `nint(zeta)` then `K**zeta_int`.
- Second product: with α stored as `(K, M)` (see 4.3), you need
  `Kmat · alphasᵀ`. That is `transb = "t"`, and the leading dimension of
  `alphas` is `K`. Line 248 of the same routine is the one-column version of
  this call. `delta**2` goes in as the scalar `alpha` argument.
- Both existing calls are guarded by `if (n_sites > 0)`. Keep the guard: a
  chunk with no atoms of the central species is normal.

### 4.3 Loading (phase 4)

A reader that sits next to `read_alphas_and_descriptors`
([read_files.f90:497](../src/read_files.f90#L497)) in the `read_files` module.

| In | Out |
|---|---|
| `file_desc`, `file_alphas`, `n_cols` (the block's `n_columns`) | `n_sparse`, `alphas_tensor(1:n_cols, 1:n_sparse)`, `Qs(1:dim, 1:n_sparse)` |

What to reuse from the existing routine:

- Counting lines of the α file gives `n_sparse`. That still works with `K`
  values per line.
- Counting lines of the descriptor file and dividing by `n_sparse` gives
  `dim`. The file is one value per line, sparse point by sparse point.
- With α stored as `(K, M)`, each line of the α file is one column:
  a single list-directed `read` of `alphas(1:n_cols, i)` per line.

Check on load: the `dim` you get from the file must equal
`soap_turbo_hypers(i)%dim`, which is only final after the compression section
ending at [read_files.f90:2677](../src/read_files.f90#L2677). Stop with a
message if not. This catches most export mistakes immediately.

---

## 5. Wiring, file by file

### 5.0 The model type and the life of a parameter

**One flag decides everything.** The `.gap` file says what kind of model it
is, through the `target` keyword of its `tensor_readout` block. A file with no
such block is an ordinary energy GAP. Store that as one integer `model_type`
with three named values (energy, dipole, polarisability), and derive a logical
`is_tensor` from it. There is no input-file keyword: the model file describes
itself, and the user runs `turbogap predict` as usual.

Fortran's idiom for an enum is a set of named integer constants
(`integer, parameter`) declared in a module, here `types`. Compare against the
names, never against bare numbers.

Who looks at which:

| Flag | Read by | Purpose |
|---|---|---|
| `is_tensor` | `read_gap_hypers`, four places in `turbogap.f90`, one in `get_gap_soap` | tensor path instead of energy path |
| `model_type` (dipole or polarisability) | the geometry routine in `tensor_methods.f90`, and the routine that derives `D` and `K` | which columns to build |
| `D`, `K` | allocation, kernel, contraction, reduce, output | sizes only |

That is the whole footprint on the central code: it asks "tensor or not" and
otherwise passes `D` and `K` around. Only your own module knows what a dipole
is.

**Every parameter, from file to use.** A parameter passes through up to five
stages: parsed on rank 0, checked, turned into derived sizes, sent to the
other ranks, used. Missing any stage for any row is a bug, and usually a
silent one.

| Parameter | `.gap` source | Stored in | Check on load | To other ranks | Used by |
|---|---|---|---|---|---|
| model type | `target` in `tensor_readout` | readout config, default energy | must be dipole or polarisability | one integer | gates; geometry |
| `r_c` | `rcut` | readout config | `> 0`; raise `rcut_max` to it | one double | geometry |
| `C` | `n_channels` | readout config | `≥ 1` | one integer | geometry |
| `C_x` | `cross_channels` | readout config | `0` for dipole, `≥ 0` for polarisability | one integer | geometry |
| species order | species list | not stored | equals the input file's `species` | — | — |
| hard-coded options | `radial`, `cutoff_function`, `species_pairs`, `unit_vector` | not stored | must equal the built-in choice (0.1) | — | — |
| `D`, `K1`, `K` | derived | readout config | — | recompute on each rank, or send | everything |
| ξ, δ | `zeta`, `delta` of each `soap_turbo` block | **existing** `soap_turbo%zeta`, `%delta` | — | **already sent** (lines 674–675) | kernel |
| sparse points | `desc_sparse` file | **existing** `soap_turbo%Qs(dim, n_sparse)` | file `dim` equals `soap_turbo%dim` | **already sent** (line 673) | kernel |
| `M_t` | line count of the α file | **existing** `soap_turbo%n_sparse` | — | **already sent** (line 559) | allocation |
| block width | `n_columns` | new `soap_turbo%n_columns`, default 0 | equals `K` | one integer per block | allocation |
| α | `alphas_sparse` file | new `soap_turbo%alphas_tensor(K, n_sparse)` | — | allocate on other ranks, then send | kernel |

The bold entries are the point. A tensor SOAP block differs from an energy
SOAP block in exactly one array: α has `K` columns instead of one. ξ, δ, the
sparse points and their count mean the same thing in both, and TurboGAP
already parses, allocates and broadcasts them. So reuse those fields, and add
only `n_columns` and `alphas_tensor`. No new per-block type is needed.

Two consequences of reusing the existing fields:

- The existing broadcast sends `Qs(1:dim, 1:n_sparse)` with `dim` taken from
  the SOAP hypers, not from your file. If the file's width differs, the
  broadcast is wrong without any error. That makes the `dim` check on load
  mandatory, not just helpful.
- The existing broadcast also sends the rank-1 `alphas(1:n_sparse)`
  unconditionally ([line 672](../src/turbogap.f90#L672)), and the other ranks
  allocate it. For a tensor block nothing fills that array on rank 0. Allocate
  it there to length `n_sparse` and zero it when you load the tensor α, so the
  existing line stays valid and you do not have to touch it.

**Derived sizes in one place.** `D`, `K1` and `K` follow from the model type,
`C`, `C_x` and the number of species (table in "Scope"). Compute them in one
small routine and call it wherever the config becomes available. Never
recompute `K` inline somewhere else; every array extent and every `dgemm`
dimension depends on it.

**Fail loudly at load.** Each "check" above should stop the run with a message
naming the keyword and both values. A model that loads has then been shown to
be one this code can evaluate, and nothing downstream needs to re-check.

### 5.1 `types.f90`

[src/types.f90](../src/types.f90). Three additions.

**The model-type constants**: three named integers, module level.

**A readout config type**, one instance per run, holding the "readout config"
rows of the table in 5.0: `model_type` (default: energy), `is_tensor`
(default `.false.`), `r_c`, `C`, `C_x`, and the derived `D`, `K1`, `K`.
Defaults matter here: for an energy `.gap` the block is absent and nothing
ever writes to these fields.

**Two fields in `soap_turbo`** ([line 60](../src/types.f90#L60)): integer
`n_columns`, default 0, and `alphas_tensor(:, :)` next to the existing
`alphas(:)`.

The `tensor_readout` keywords from guide 7.7, for reference:

| `.gap` keyword | Use |
|---|---|
| `target` | read: sets `model_type` |
| species order | check against the input file |
| `rcut` | read: `r_c` |
| `n_channels` | read: `C` |
| `cross_channels` | read: `C_x` |
| `radial` | check: warped Chebyshev |
| `cutoff_function` | check: cosine |
| `species_pairs` | check: on |
| `unit_vector` | check: true |
| `cross_l_max` | ignore (3b only) |

If you add a `write_property` slot for the output, the array is declared
`(1:11)` at [line 174](../src/types.f90#L174).

### 5.2 `read_files.f90`

`read_input_file` is not touched.

**`.gap` parsing.** `read_gap_hypers`
([line 2161](../src/read_files.f90#L2161)). The file is read twice:

1. First pass ([2214–2236](../src/read_files.f90#L2214)): counts
   `gap_beg <type>` lines per type, then allocates.
2. Second pass ([2258–2877](../src/read_files.f90#L2258)): one big
   `if / else if` on the block type, each with its own keyword loop that runs
   until `gap_end`.

Every keyword line has the form `keyword = value` with spaces around `=`. The
parser reads the keyword, backspaces, then reads `cjunk, cjunk, value`.

The file layout is fixed by guide 7.7: a `tensor_readout` block, then one
`soap_turbo` block per central species (and per scale), each carrying `zeta`,
`delta`, `n_columns`, `desc_sparse`, `alphas_sparse`.

What to add:

| Where | What |
|---|---|
| first pass | nothing: `tensor_readout` is a single block and needs no count |
| `soap_turbo` keyword chain, [2307–2541](../src/read_files.f90#L2307) | **one** new branch: `n_columns`. `zeta`, `delta`, `desc_sparse`, `alphas_sparse` are already parsed there |
| the read at [2552–2558](../src/read_files.f90#L2552) | branch on `is_tensor` (see below) |
| new `else if` next to `distance_2b` at [2679](../src/read_files.f90#L2679) | the `tensor_readout` block: a keyword loop like the `distance_2b` one |
| argument list | one new `intent(out)` argument for the config; update the single call at [turbogap.f90:466](../src/turbogap.f90#L466) |

Three details:

- **The `tensor_readout` block must come before any `soap_turbo` block**, as
  guide 7.7 has it. Then `is_tensor` and `K` are known by the time a SOAP
  block is parsed. Enforce the order: if a `soap_turbo` block has
  `n_columns > 0` while the model type is still energy, stop with an error.
  That one check also catches a missing `tensor_readout` block.
- **The read at [line 2552](../src/read_files.f90#L2552)** is currently
  `if (do_prediction)` with the rank-1 reader. Make it a two-way choice with
  the tensor case first: if `is_tensor`, call your 2D reader into the
  existing `n_sparse` and `Qs` and the new `alphas_tensor`, and allocate and
  zero the rank-1 `alphas` (5.0); otherwise the existing code, unchanged. The
  order matters because `do_prediction` is true here in a normal
  `turbogap predict` run, and the rank-1 reader would accept your `K`-column
  file without complaint (0.3).
- Check `n_columns` against `K` right there. Check the file's `dim` against
  `soap_turbo%dim` after the compression section (ends at
  [2677](../src/read_files.f90#L2677)), since that is where `dim` is set.
- `rcut_max` is raised per block
  ([2598–2602](../src/read_files.f90#L2598)). Do the same with the readout
  `r_c`. In practice it equals the SOAP cutoff, so this is a safeguard.

### 5.3 `gap_interface.f90` — the SOAP hook

`get_gap_soap` ([line 38](../src/gap_interface.f90#L38)). Called from exactly
one place, [turbogap.f90:1586](../src/turbogap.f90#L1586).

**Where.** After the `get_soap` call
([241–247](../src/gap_interface.f90#L241)), next to the local-property block
([253–303](../src/gap_interface.f90#L253)), under `is_tensor`. It must
not be inside `if (do_prediction)`
([line 305](../src/gap_interface.f90#L305)).

**New arguments.** Four, because the reused fields are already there:

| Argument | Intent | Shape |
|---|---|---|
| `is_tensor` | in | scalar logical |
| `alphas_tensor` | in | `(1:K, 1:n_sparse)` |
| `v` for this chunk | in | `(1:D, 1:K, 1:this_n_sites_mpi)` |
| tensor accumulator | inout | `(1:D)` |

`Qs`, `delta`, `zeta` and `this_i_beg` are existing arguments and carry the
right values for a tensor block. The routine does not need the model type:
`D` and `K` are the extents of `v`.

One thing to know about `alphas_tensor` in an energy run: it is then
unallocated in the caller. Passing an unallocated array to an ordinary
assumed-shape dummy is not allowed. Either declare the dummy `allocatable`
(as `local_property_models` is, line 58), or allocate it to size zero for
energy blocks.

**The index mapping.** This is the part that is easy to get wrong.

- The caller loops over chunks `this_i_beg : this_i_end` of the rank's atoms
  ([turbogap.f90:1537–1542](../src/turbogap.f90#L1537)).
- Inside, only atoms of the central species are kept. Column `i` of `soap` is
  kept site `i`.
- `in_to_out_site(i)` is that site's **global** atom id
  ([gap_interface.f90:177–190](../src/gap_interface.f90#L177)). The existing
  code relies on this at line 282, where it indexes a global-size array.

So if the caller passes `v(:, :, this_i_beg:this_i_end)`, the dummy starts at
1 and the geometry of kept site `i` is at third index
`in_to_out_site(i) − this_i_beg + 1`.

The local-property loop at
[280–285](../src/gap_interface.f90#L280) is the same mapping, written for a
global-size target.

**Accumulating.** The routine is called once per chunk and per block, so the
accumulator must be `intent(inout)` and only ever added to. Note that
`virial` is `intent(out)` in this routine and the caller re-sums it; do not
copy that pattern for the tensor.

### 5.4 `turbogap.f90`

In execution order.

**Declarations** (around [line 167](../src/turbogap.f90#L167)): the readout
config, `v`, the accumulator and a second copy of it for the MPI reduce. Add
`use tensor_methods` to the list at the top.

**Read and broadcast** ([466](../src/turbogap.f90#L466),
[523–805](../src/turbogap.f90#L523)): see section 6.

**Turn the energy path off**, straight after the `#endif` at
[line 805](../src/turbogap.f90#L805), so that it runs on every rank and in the
serial build: if `is_tensor`, set `params%do_prediction`, `params%do_forces`
and `params%do_derivatives` to `.false.`. `params` is a per-rank copy, which
is why this must happen after the model type has been broadcast and not only
on rank 0. From here on the existing code skips energies, forces, 2b, 3b and
vdW by its own tests, and you have not edited any of them.

**Main condition** ([line 1305](../src/turbogap.f90#L1305)): add `is_tensor`
to the `.or.` chain. Without this a tensor run falls into the
"You didn't ask me to do anything!" branch at line 3425.

**Zero the accumulator** with the energies at
[1360](../src/turbogap.f90#L1360).

**Geometry call**, between [1521 and 1523](../src/turbogap.f90#L1521), under
`is_tensor`. Pass the slices exactly as the 2b energy call does at
[2927–2932](../src/turbogap.f90#L2927):

| Actual argument |
|---|
| `n_neigh(i_beg:i_end)` |
| `rjs(j_beg:j_end)` |
| `xyz(1:3, j_beg:j_end)` |
| `neighbor_species(j_beg:j_end)` |

Allocate `v` with the site index running over `i_beg:i_end`. The size changes
between frames; follow the `n_sites /= n_sites_prev` reallocation pattern at
[line 1310](../src/turbogap.f90#L1310), or simply deallocate and reallocate
every frame.

**SOAP loop**: extend the call at
[1586](../src/turbogap.f90#L1586) with the arguments of 5.3. Pass
`v(:, :, this_i_beg:this_i_end)`.

**Reduce**: after the SOAP loop ends at
[1794](../src/turbogap.f90#L1794). Copy the form of
[line 1517](../src/turbogap.f90#L1517): `mpi_reduce` of `D` doubles with
`MPI_SUM` to rank 0, inside `#ifdef _MPIF90`. Only rank 0 writes, so no
broadcast back is needed. In the serial build the accumulator is already the
total.

**Output** ([3394–3424](../src/turbogap.f90#L3394)): extend the condition
with `is_tensor` and pass the tensor to the writer (5.5). The final-timing
condition at [line 4790](../src/turbogap.f90#L4790) is cosmetic.

### 5.5 `xyz.f90`

`write_extxyz` ([line 129](../src/xyz.f90#L129)) has eight call sites in
`turbogap.f90` (3411, 3662, 3684, 4182, 4200, 4355, 4372, 4488). To avoid
editing all of them, add the new arguments at the **end** of the list and
declare them `optional`; test with `present(...)`. The routine is in a module,
so the explicit interface that optional arguments require is already there.

The header fields are written one by one with `advance="no"`. The virial
block at [315–322](../src/xyz.f90#L315) is the pattern for a quoted list of
numbers. Two points:

- `F16.8` is too coarse for a 1e-10 comparison against Python. The SOAP dump
  at [turbogap.f90:1741](../src/turbogap.f90#L1741) uses `ES24.15`; use
  something like that while validating.
- The extra energy `string` is only written inside the
  `write_property(7)` branch ([306–312](../src/xyz.f90#L306)). Do not hide
  your field in there.

For ASE: a quoted 3-vector is parsed to an array of 3, and a quoted list of 9
to a 3×3 matrix. Six Voigt numbers come back as a flat array of 6. That
decides open question 3 of the plan for you one way or the other.

### 5.6 `Makefile`

[Makefile](../Makefile), the `SRC` list at line 26. Add `tensor_methods.f90`
after `types.f90` and `neighbors.f90`, and before `gap_interface.f90`, which
will `use` it. Order matters: the `.mod` file must exist before a dependent
file compiles.

For the phase 1 standalone driver: `PROGRAMS := turbogap` plus the rule
`$(BIN_DIR)/%: src/%.f90` means that a `src/<name>.f90` containing a
`program`, with `<name>` added to `PROGRAMS`, is built and linked against
every module automatically.

The active makefile
([makefiles/Makefile.Ubuntu_gfortran_mpi](../makefiles/Makefile.Ubuntu_gfortran_mpi))
compiles with `-O0 -fcheck=all -Wall -fbacktrace`. Out-of-bounds indices and
shape mismatches stop the run with a line number, which is what you want
while porting index formulas.

---

## 6. MPI: getting the model to the other ranks

`read_gap_hypers` runs on rank 0 only. Everything it produced is then sent
field by field in [turbogap.f90:523–805](../src/turbogap.f90#L523). The
existing sequence is:

1. broadcast the block counts ([525–528](../src/turbogap.f90#L525));
2. rank 0 packs the sizes into temporary integer arrays, broadcasts them
   ([557–626](../src/turbogap.f90#L557));
3. other ranks allocate with the helpers in
   [mpi.f90](../src/mpi.f90) ([629–639](../src/turbogap.f90#L629));
4. broadcast every array and scalar
   ([654–796](../src/turbogap.f90#L654)).

`params` is not part of this: every rank reads the input file itself.

Because the tensor blocks reuse the existing fields (5.0), almost everything
is already sent. What you add:

- **Readout config**: `model_type`, `r_c`, `C`, `C_x`. Four scalar
  broadcasts, anywhere after line 532. Then derive `is_tensor`, `D`, `K1`,
  `K` on every rank with the routine of 5.0, rather than sending them.
- **Per block**, inside the loop at [654–740](../src/turbogap.f90#L654):
  broadcast `n_columns`; if `is_tensor`, then
  `if (rank /= 0) allocate` `alphas_tensor(n_columns, n_sparse)`, then
  broadcast it. `n_sparse` is already known on every rank at that point
  (line 669).

`mpi.f90` stays unchanged. The plan lists it as changed; this route does not
need it.

What you rely on, and should not break: `n_sparse`, `Qs`, the rank-1
`alphas`, `delta` and `zeta` are sent by the existing lines 559 and 672–675.
That only works if rank 0 has set `n_sparse`, filled `Qs` with the SOAP
hypers' `dim`, and allocated the rank-1 `alphas` (5.0).

Things to copy exactly:

- 2D arrays: count is the product of the extents
  ([line 673](../src/turbogap.f90#L673)).
- The whole section is inside `#ifdef _MPIF90`. A serial build must still
  compile.

A wrong count or a missed field does not crash. It gives a wrong answer on
more than one rank only. That is why the plan's phase 5 check runs serial and
2–4 ranks.

---

## 7. A tensor-only run

The user runs `turbogap predict` with an unchanged input file and a tensor
`.gap`. These are all the places where `is_tensor` is tested:

| Place | Effect |
|---|---|
| [read_files.f90:2552](../src/read_files.f90#L2552) | 2D reader instead of the rank-1 reader |
| after [turbogap.f90:805](../src/turbogap.f90#L805) | `do_prediction`, `do_forces`, `do_derivatives` forced off |
| [turbogap.f90:1305](../src/turbogap.f90#L1305) | main block entered |
| [turbogap.f90:1521](../src/turbogap.f90#L1521) | geometry computed |
| in `get_gap_soap` | kernel and contraction run; energies skipped because `do_prediction` is now false |
| [turbogap.f90:3394](../src/turbogap.f90#L3394) | output written |

`turbogap predict` sets `do_prediction` and `do_forces` to true before the
input file is read
([read_files.f90:665–667](../src/read_files.f90#L665)). The override after
line 805 undoes that. Note the one place that still sees the original value:
`read_gap_hypers` runs before the override, which is why the tensor case must
be tested first there (5.2).

One thing I could not verify without running it: with `do_forces` false,
`forces` is never allocated (it is allocated at
[turbogap.f90:1430](../src/turbogap.f90#L1430) only under `do_forces`), yet
it is passed to `write_extxyz`. Under `-fcheck=all`, gfortran can abort with
"Allocatable actual argument 'forces' is not allocated". If you see that
message, this is the cause.

---

## 8. Phase 0 on the Python side

The target file layout is guide 7.7. The functions it names are not in the
checkout (0.2), so this section lists where the data for them lives.

**Config for a model in this scope** (0.1):

| Field | dipole | polarisability |
|---|---|---|
| `target_mode` | 1 | 2 |
| `gap_site_models` | `["soap"]` | `["soap"]` |
| `dipole_unit_vector` | `True` | `True` |
| `dipole_cutoff` | `"cosine"` | `"cosine"` |
| `dipole_radial` | `"chebyshev_warped"` | `"chebyshev_warped"` |
| `dipole_species_pairs` | `"symmetric"` | `"symmetric"` |
| `gap_cross_channels` | 0 | `C_x` (4 in production) |
| `gap_extra_rcut` | `None` | `None`, or 6.0 for the second scale |

**Model export.** After `fit_gap_dipole` returns `model`:

| Fortran block | Sparse points | α |
|---|---|---|
| SOAP, species `t` | `model.site_sparse["soap"][t]`, `[M_t, Q]` | `model.site_alpha["soap"][t]`, flat `[M_t·K]` |
| second scale, species `t` | `model.sparse_desc2[t]` | `model.alpha2[t]` |

- Sparse file: flatten row by row, one value per line. That is the layout
  the line-counting in `read_alphas_and_descriptors` expects.
- α file: reshape to `[M_t, K]` and write one row per line. The flat order is
  `(m, k)` with `m` major (`_phi_frame` docstring), so a plain reshape is
  correct.
- Write at 17 significant digits.

**`.gap` values for the SOAP blocks.**

| `.gap` keyword | Python |
|---|---|
| `rcut` | `rcut_hard` (second scale: `gap_extra_rcut`) |
| `buffer` | `rcut_hard − rcut_soft` (the parser converts it back, [read_files.f90:2543](../src/read_files.f90#L2543)) |
| `n_max` | `alpha_max`, once per species |
| `l_max`, `atom_sigma_r`, `atom_sigma_t`, `…_scaling`, `amplitude_scaling`, `central_weight`, `radial_enhancement` | same names |
| `basis`, `scaling_mode` | same names |
| `compress_soap = .true.`, `compress_mode = trivial` | `compress_mode = "trivial"` |
| `central_species` | `t + 1` |
| `species` | `O H`, matching `types = [8, 1]` |
| `zeta`, `delta`, `n_columns` | ξ, `gap_delta_soap`, `K` |

Production SOAP is `l_max = 7`, `alpha_max = 7` with trivial compression,
which gives the width 216 quoted in the guide. The second scale keeps the
same soft/hard gap (`gap_dipole.py:442`), so its `buffer` equals the first
block's.

**Reference dumps.** `_predict_numpy` (`gap_dipole.py:567`) already holds
every intermediate:

| Quantity | Where |
|---|---|
| `v_i` | `vs[f]`, `[N, D, K]` |
| `V_i` before the cross group | `tnep_data._neighbor_dipole_V(...)`, `[N, D, G·C, T]`; apply `_collapse_V` to get the column order of 3.2 |
| SOAP `q_i` | `site["soap"][0][f]` |
| second-scale `q_i` | `descs2[f]` |
| site scalars | not formed in Python. Compute per block as `dm**2 * (D[sel] @ Dm.T)**xi @ alpha.reshape(M, K)` |
| final μ / χ | the return value |

Dump with the Python index order and note the shape in a header line. When
reading in Fortran, remember that Python `[N, D, K]` written row-major is
Fortran `(K, D, N)` read in natural order, not `(D, K, N)`.

**Units.** With `dipole_units = "e*bohr"` and
`convert_dipole_to_eangstrom = False` the model predicts in the units of the
training labels. Nothing in the forward pass converts. This is open
question 1 of the plan.

---

## 9. Checks that isolate one thing at a time

Beyond the per-phase checks in the plan:

| Check | Isolates |
|---|---|
| Rotate the frame, compare μ (rotates) and the trace of χ (invariant) | sign and Voigt-order errors, no Python needed |
| With only the isotropic group's α non-zero, χ must be a multiple of `(1,1,1,0,0,0)` | group and column ordering |
| Translate all atoms by a constant | neighbour-list handling |
| One SOAP block only, one species only | block bookkeeping; `test_species_block_isolation` in `TNEP/tests/test_gap_dipole.py` is the Python twin |
| `write_soap = .true.` with one MPI task; `soap_<block>.dat` has a header `n_sites n_soap`, then one row per atom, zeros for other species | SOAP parity (phase 4), before any tensor code runs |
| Feed dumped `q` and `v` through kernel + contraction only | readout, independent of SOAP |
| Serial against 2 and 3 ranks, with an atom count not divisible by the rank count | broadcast and chunk indexing |
| Lower `max_Gbytes_per_process` in the input until the SOAP loop takes several chunks | the `this_i_beg` offset of 5.3 |

`write_soap` refuses to run on more than one MPI task
([turbogap.f90:428](../src/turbogap.f90#L428)), so the SOAP comparison is
serial only.

---

## 10. Order of work, mapped to the plan

| Plan phase | Sections here | Guide |
|---|---|---|
| 0 Python | 8 | 7.7 |
| 1 Geometry | 2.1, 3.1–3.3, 4.1, 5.6 (driver) | 3.1–3.5 |
| 2 2b / 3b | out of scope | — |
| 3 Kernel and contraction | 3.4, 4.2 | 3.9 |
| 4 Loading | 4.3, 5.1, 5.2, 6 | 2, 7.7 |
| 5 Wiring | 5.3, 5.4, 5.5, 7 | — |
| 6 Tests | 9; `tests/` currently holds only `stopping/` | 7.6 |

Target widths to check first, with `C = 10`, `T = 2`: dipole `K = 20`;
polarisability with `C_x = 4`: `K = 40 + 36 = 76`.
