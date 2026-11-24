# Dipole and polarisability prediction — implementation plan

**Goal:** evaluate a fitted TNEP GAP dipole (μ, 3 components) or polarisability
(χ, 6 Voigt components) model on each frame TurboGAP reads, and write the
result to the output xyz.

**Reference:** `TNEP/docs/gap_forward_pass.md` (called "the guide" below;
section numbers refer to it). The Python oracle is `gap_dipole.predict`.

**Scope:** forward pass only. No forces, no derivatives, no fitting.

---

## 1. The idea in one paragraph

For every atom, each site model turns a unit-norm descriptor into `K` signed
scalars through the usual GAP kernel: `s_i = δ² · (q_i · Q_sparse)^ζ · α`.
Those scalars are contracted with a per-atom geometric tensor built from the
neighbour list (`v_i`, or `W_i` for the 3b model), and the result is summed
over atoms. The contraction is linear, so every block (SOAP per species,
second SOAP scale, 2b, 3b) adds its own contribution to the frame total
independently. They only share the geometry.

## 2. Where each piece lives

| Piece | File | New or changed |
|---|---|---|
| All new maths (radial basis, geometry, site descriptors, kernel, contraction) | `src/tensor_methods.f90` | new (file exists, empty) |
| Model and config data structures | `src/types.f90` | changed |
| `.gap` keywords, 2D α reader | `src/read_files.f90` | changed |
| Allocation on non-root ranks | `src/mpi.f90` | changed |
| SOAP site-model hook | `src/gap_interface.f90` (`get_gap_soap`) | changed |
| Geometry call, 2b/3b block loop, MPI broadcast and reduce | `src/turbogap.f90` | changed |
| `dipole=` / `polarizability=` header field | `src/xyz.f90` (`write_extxyz`) | changed |
| Build | `Makefile` (`SRC` list, after `neighbors.f90`) | changed |
| Model exporter and reference dumps | TNEP repo (Python) | new |
| Regression test | `tests/tensor_readout/` | new |

## 3. What is reused as is

- Frame reading and the multi-frame loop (`read_xyz`, main loop).
- The neighbour list from `build_neighbors_list`: `rjs`, `xyz`, `n_neigh`,
  `neighbor_species`. It is already centre-sorted, both orders, all images.
- The SOAP descriptor from `get_soap`, already unit-normalised and compressed,
  one `soap_turbo` block per central species. The 6 Å second scale is simply
  one more `soap_turbo` block.
- The sparse descriptor file layout (one value per line, point-major).
- The MPI site split (`i_beg:i_end`) and the reduce pattern used for energies.

## 4. Decisions already made

1. **One module for the maths.** Everything new and numerical goes in
   `tensor_methods.f90`; the other files only get wiring.
2. **Geometry is computed once per frame per rank**, straight after the
   neighbour list, and passed to the blocks that need it. It is not recomputed
   inside each `soap_turbo` block.
3. **The readout cutoff is its own parameter** (`r_c`), not inherited from the
   SOAP block. The second SOAP scale (6 Å) still reads out through the 4 Å `v_i`.
4. **The 2b/3b site models are new block types.** The existing `distance_2b`
   and `angle_3b` are energy GAPs with different descriptors and kernels and
   are not touched.
5. **`local_property_predict` is not modified.** It handles one α column, adds
   `V0` and clamps to ≥ 0. A separate multi-column, unclamped routine is added
   instead, so Hirshfeld and XPS behaviour cannot change.
6. **A separate switch** (for example `do_tensor_prediction`) loads and runs
   the model, so a dipole-only `.gap` file does not need energy α files.

## 5. Conventions that will bite

| Topic | TurboGAP | Guide | Action |
|---|---|---|---|
| Bond vector | `xyz = r_j − r_i` | `d_ij = r_i − r_j` | negate on entry; matters for every dipole term |
| Self entry | first neighbour of each atom is itself, `r = 0` | self dropped | skip the first entry |
| Cutoff | list built at global max cutoff + buffer | pairs with `r < r_c` only | filter |
| Indexing | 1-based, column-major | 0-based, row-major | column orders of guide section 4 must be kept exactly; store α as `(K_m, M_t)` so a row-major text file reads straight in |
| Species index | order of `species` in the input file | `types = [8, 1]` (O, H) | require the same order, stop with an error otherwise |
| SOAP components | set by the `soap_turbo` hypers and `compress_mode` | whatever quippy produced | validate `q_i` first (phase 4) |

---

## 6. Phases

Each phase ends with a check that can fail. Do not start the next phase until
it passes. Target agreement with the Python reference is about 1e-10 relative.

### Phase 0 — Python side (TNEP repo)

- [ ] Exporter: for each block write `sparse[m][t]` (QUIP sparseX layout) and
      `alpha[m][t]` (`K_m` values per line, one line per sparse point), plus a
      `.gap` file with the section-5 fields and the SOAP hypers.
- [ ] Reference dumps for 3–5 water frames, both production models:
      `q_i` per SOAP scale, `V_i`, `v_i`, `W_i`, `q2b_i`, `q3b_i`,
      per-block site scalars, final μ / χ. Plain text, full precision.

**Check:** reloading the exported files in Python reproduces `gap_dipole.predict`.

### Phase 1 — Geometry in `tensor_methods.f90`

Pure routines taking the neighbour arrays, no dependence on the rest of the code.

- [ ] Radial basis: warped Chebyshev × cutoff (`cosine`, `cosine2`), evaluated
      once at `max(C, C_x, C_s)` channels (guide 3.2).
- [ ] Bond basis `V_i` with species-pair columns, rank 1 and rank 2, with the
      unit-vector option (3.3, 3.4).
- [ ] Cross group, polarisability form and dipole form; assemble
      `v_i = [V_i | cross]` (3.5).
- [ ] A small standalone driver or test program that reads a frame, calls
      these and prints the arrays.

**Check:** `V_i` and `v_i` match the dumps for both models. Shapes: dipole
`K = 36`, polarisability `K = 76`.

### Phase 2 — 2b and 3b pieces in `tensor_methods.f90`

- [ ] 2b site descriptor, unit-normalised, zero rows left zero (3.7).
- [ ] Legendre recurrence and the triplet loop over ordered neighbour pairs.
- [ ] 3b site descriptor (3.8).
- [ ] 3b angular kernel `W_i`, polarisability form (3.9). Uses the full `d`
      even when the unit-vector flag is on.

**Check:** `q2b_i`, `q3b_i`, `W_i` match the dumps. Widths 20, 216, 144.

The dipole form of the 3b kernel is not in production; skip it until needed.

### Phase 3 — Kernel and contraction in `tensor_methods.f90`

- [ ] Site scalars: `s = δ² · (qᵀ Q)^ζ · α` as two `dgemm` calls, `K_m`
      columns, no clamp, no offset.
- [ ] Contraction `y += Σ_k s_i[k] · v_i[:, k]`, accumulated over atoms.

**Check:** feeding the dumped Python descriptors and geometry through these
two routines reproduces the dumped site scalars and final μ / χ. This isolates
the readout from any SOAP mismatch.

### Phase 4 — Model loading

- [ ] `types.f90`: readout config type; 2D `alphas` on the per-block model;
      block type for 2b/3b site models.
- [ ] `read_files.f90`: new keywords in `read_gap_hypers`, new block types,
      2D α branch in `read_alphas_and_descriptors`, the new input-file switch.
- [ ] `mpi.f90` and the broadcast section of `turbogap.f90`: allocate and
      broadcast the new fields.
- [ ] Write the SOAP descriptor with the existing `write_soap` option and
      compare it to the dumped `q_i`.

**Check:** `q_i` from TurboGAP matches the Python builder for both scales.
This is the step most likely to need iteration; settle it before wiring.

### Phase 5 — Wiring in the main program

- [ ] Call the geometry routines after `build_neighbors_list` for this rank's
      sites.
- [ ] In `get_gap_soap`, next to the existing local-property prediction: site
      scalars and contraction for SOAP blocks that carry a tensor model.
- [ ] Next to the existing 2b/3b loop: evaluate the new 2b/3b site blocks.
- [ ] `mpi_reduce` the 3 or 6 numbers.
- [ ] `write_extxyz`: new header field, new `write_property` slot.

**Check:** μ and χ in the output xyz match the Python reference on the dumped
frames, serial and with 2–4 MPI ranks.

### Phase 6 — Regression test and docs

- [ ] `tests/tensor_readout/`: one small frame, the exported model files, the
      expected output, and a script that runs TurboGAP and compares.
- [ ] Short section in the README or docs: keywords and the exporter.

---

## 7. Out of scope

Forces and derivatives, the Ewald field channel, the other radial families
(Gaussian, Bessel, screened, plain Chebyshev), the 3b dipole kernel, and any
performance work beyond a compiled triplet loop. The descriptor dominates the
cost (guide section 7), so nothing here needs optimising first.

## 8. Open questions

1. Units in the output: the guide says μ is in e·a₀ and χ in Å³ with no
   conversion. Keep as is, or convert μ to e·Å on output?
2. Should per-atom contributions also be written as array properties, or only
   the frame total?
3. Name of the header fields: `dipole` / `polarizability`, or something that
   matches an existing convention in your analysis scripts?
