# GAP forward pass — porting guide for the two production models

What a Fortran implementation has to compute to evaluate the production TNEP GAP models
on a frame: the dipole model (SOAP + 2b site model + dipole cross group, 1.47 % test RRMSE on
bulk water) and the polarisability model (SOAP at two scales + 3b site model + polarisability
cross group, 13.0 / 13.9 %). Fitting is out of scope; a model is fixed tensors plus the
constants below. The SOAP-turbo descriptor is an input: one unit-normalised vector per atom
from the existing implementation. Everything here is read off the NumPy reference path
`gap_dipole._predict_numpy`, which the fit also uses, so fit and prediction cannot disagree.

Notation: `N` atoms, `T = 2` species in the order `types = [8, 1]` (O, H), `t_i` the type index
of atom i, `D` = 3 (dipole) or 6 (polarisability, Voigt `xx yy zz xy yz zx`), `M_t` sparse points
of species t, `Q = 216` the SOAP width. Kernel power ξ = 2 throughout. All arithmetic float64.

---

## 1. The two models

| | dipole | polarisability |
|---|---|---|
| target | μ [3], e·a₀ (no conversion) | χ = V(ε−1)/4π [6 Voigt], Å³ |
| readout and SOAP cutoff r_c | 3.5 Å | 4.0 Å |
| cutoff function f_c | cosine² | cosine |
| bond vector in the readout | full d | unit û |
| radial channels C | 10 | 10 |
| site models (own sparse set and α each) | soap (250 / species), 2b (40 / species) | soap (250), 3b (40), soap at 6.0 Å (250) |
| δ per block (enters squared) | soap 1.0, 2b 0.1 | soap 1.0, 3b 0.1, soap-6Å 1.0 |
| cross group channels C_x | 2 | 4, Legendre order L = 3 |
| 2b / 3b descriptor channels C_s, order L_s | 10 | 6, 1 |

Exact settings: `diagnostics/gap_experiments/gap_scan.py` (`MU`, `POL`, `ROWS`).

---

## 2. Model tensors

Per block `m` (dipole: soap, 2b; polarisability: soap, 3b, soap6) and species t:

| tensor | shape | meaning |
|---|---|---|
| `sparse[m][t]` | `[M_t, Q_m]` | unit-norm sparse descriptors of species t |
| `alpha[m][t]` | `[M_t, K_m]` | coefficients (stored flat row-major, `k` fastest) |
| `δ_m` | scalar | block amplitude, used as δ_m² |

`Q_m`: 216 for the SOAP blocks, 20 for 2b, 216 for 3b (section 3.7). `K_m`: the readout width
the block reads through, 36 (dipole) / 76 (polarisability) for soap, 2b and soap6, 144 for 3b.

A frame needs positions, the (orthorhombic, periodic) cell, atomic numbers → `types.index(Z)`.

---

## 3. The forward pass

### 3.1 Neighbour pairs

All pairs `(i, j)` with `|r_i − r_j + L| < r_c`, periodic images included, `i == j` excluded, both
orders present. Bond vector

    d_ij = r_i − r_j        (neighbour → centre)

Water cells (L ≈ 12.4 Å) satisfy `L ≥ 2 r_c`, so minimum-image over all atom pairs is exact.
Order the list by centre; every readout below is a segmented sum over the centre i.

### 3.2 Radial basis

With `r = |d|` and `x = 2 (r/r_c − 1)² − 1`:

    R_n(r) = T_n(x) · f_c(r),   n = 0 … C−1,   T_0 = 1, T_1 = x, T_{n+1} = 2x T_n − T_{n−1}

    f_c(r) = ((1 + cos(π r / r_c)) / 2)²    dipole
    f_c(r) = (1 + cos(π r / r_c)) / 2       polarisability

One recurrence serves every radial weight in the model; the bond basis uses C = 10 channels,
the cross group C_x (2 or 4), the 2b/3b descriptors C_s (10 or 6).

### 3.3 Bond geometry g(d)

Dipole: `g(d) = d`, one group (`G = 1`), `D = 3`.

Polarisability: with û = d/|d|, two Voigt groups (`G = 2`), `D = 6`:

    group 0:  (1, 1, 1, 0, 0, 0)
    group 1:  (û_x², û_y², û_z², û_x û_y, û_y û_z, û_z û_x) − group0 / 3        (= ûûᵀ − I/3)

û is used in the bond basis (3.4) and the cross vectors (3.5) of the polarisability model only;
the 3b angular kernel (3.8) uses the full d.

### 3.4 Bond basis V

    V_i[:, g·C + n, t′] = Σ_{j : t_j = t′}  R_n(r_ij) · g_group(d_ij)

flattened channel-major, neighbour type fastest:

    column  c = (g·C + n) · T + t′,      K_V = G·C·T  = 20 (dipole), 40 (polarisability)

### 3.5 Cross group, appended after the bond basis

Cross vectors with C_x channels, same flattening:

    v_i[:, a·T + t′] = Σ_{j∈t′} R_a(r_ij) · d_ij  (dipole, full d)  /  · û_ij  (polarisability),    K₁ = C_x·T

**Polarisability** (K₁ = 8): for every channel pair `p ≤ q` (upper triangle enumerated row by row:
(0,0), (0,1), …, (0,7), (1,1), …) one Voigt column

    S = ½ (v^p v^qᵀ + v^q v^pᵀ),   S ← S − tr(S)/3 · I,   → (xx, yy, zz, xy, yz, zx)

`K_x = K₁(K₁+1)/2 = 36`.

**Dipole** (K₁ = 4): with `T^a_i = Σ_j R_a(r_ij) d_ij d_ijᵀ` (3×3), one vector column per ordered `(a, b)`

    w_i^{(a,b)} = T^a_i · v^b_i,      column = a·K₁ + b,      K_x = K₁² = 16

Readout vector of atom i:

    v_i [D, K] = [ V_i | cross ],   K = K_V + K_x = 36 (dipole), 76 (polarisability)

### 3.6 SOAP descriptors (input)

`q_i`: the unit-normalised soap_turbo vector of atom i at r_c (width 216). The polarisability
model also takes a second one at 6.0 Å (`q6_i`, same width, own sparse set and α, same readout
`v_i`). Never renormalise; the sparse points are vectors of the same kind.

### 3.7 2b and 3b site descriptors

**2b** (dipole), C_s = 10:

    q_i[t′, n] = Σ_{j∈t′} R_n(r_ij),    flatten (t′, n) → width T·C_s = 20,   q_i ← q_i / ‖q_i‖

**3b** (polarisability), C_s = 6, L_s = 1. Over ordered neighbour pairs `j ≠ k` of i (both orders),
`cos θ = d_ij · d_ik / (r_ij r_ik)`, Legendre `P_0 = 1, P_1 = cos θ`, and the unordered
species-pair slot `S(t_j, t_k)`: (O,O)→0, (O,H)→1, (H,H)→2:

    q_i[S, n, n′, l] = Σ_{j≠k} R_n(r_ij) R_n′(r_ik) P_l(cos θ_jik)

flatten `(S, n, n′, l)` → width `3·36·2 = 216`, then `q_i ← q_i / ‖q_i‖`. Zero rows stay zero.

### 3.8 3b angular kernel (what the 3b block reads out through)

Over the same `j ≠ k` triplets, with the species-resolved cross-channel weights
`R_p(j) = R_a(r_ij)` placed in slot `p = a·T + t_j` (K₁ = 8), Legendre up to L = 3, full d:

    W_i[:, (p,q), l] = Σ_{j≠k} ½ (R_p(j) R_q(k) + R_q(j) R_p(k)) · P_l(cos θ) · [½(d_ij d_ikᵀ + d_ik d_ijᵀ) − tr/3 · I]_Voigt

for `p ≤ q` (same triangle enumeration as 3.5) and `l = 0 … 3`; column `= c·(L+1) + l`,
`K_x3 = 36·4 = 144`. A product `R_p(j) R_q(k)` is non-zero only when p, q name the types of j, k.
Legendre by `(l+1) P_{l+1} = (2l+1) x P_l − l P_{l−1}`.

### 3.9 Kernel, site scalars, readout

For each block m and atom i of species t, with that block's descriptor `q_i` (q, q6, 2b or 3b):

    s_i^{(m)}[k] = δ_m² · Σ_{m′} alpha[m][t][m′, k] · (q_i · sparse[m][t][m′])²

Then

    dipole:          μ = Σ_i Σ_{k<36}  ( s^soap_i[k] + s^2b_i[k] ) · v_i[:, k]
    polarisability:  χ = Σ_i [ Σ_{k<76} ( s^soap_i[k] + s^soap6_i[k] ) · v_i[:, k]  +  Σ_{c<144} s^3b_i[c] · W_i[:, c] ]

No offsets, no mean, no unit conversion, no post-processing.

---

## 4. Index conventions

| object | size | order |
|---|---|---|
| Voigt | 6 | xx, yy, zz, xy, yz, zx |
| bond-basis column | `G·C·T` | `(g·C + n)·T + t′` |
| cross vector slot | `K₁ = C_x·T` | `a·T + t′` |
| channel pair (polarisability cross, 3b kernel) | `K₁(K₁+1)/2` | upper triangle `p ≤ q`, row by row |
| dipole cross column | `K₁²` | `a·K₁ + b` |
| 3b kernel column | 144 | `c·(L+1) + l` |
| 2b descriptor | `T·C_s` | `t′·C_s + n` |
| 3b descriptor | `3·C_s²·(L_s+1)` | `((S·C_s + n)·C_s + n′)·(L_s+1) + l` |
| species-pair slot S | 3 | (O,O)=0, (O,H)=1, (H,H)=2 |
| `alpha[m][t]` | `[M_t, K_m]` | row = sparse point, column = readout column |

---

## 5. Shapes

Dipole (`D = 3`):

| block | sparse / species | Q | K | α per species |
|---|---|---|---|---|
| soap | 250 | 216 | 36 (20 + 16) | [250, 36] |
| 2b | 40 | 20 | 36 | [40, 36] |

Polarisability (`D = 6`):

| block | sparse / species | Q | K | α per species |
|---|---|---|---|---|
| soap (4.0 Å) | 250 | 216 | 76 (40 + 36) | [250, 76] |
| 3b | 40 | 216 | 144 | [40, 144] |
| soap (6.0 Å) | 250 | 216 | 76 | [250, 76] |

Model size: 0.6 MB (dipole), 1.4 MB (polarisability) in float64.

---

## 6. Profile of the Python forward pass

50 test frames of bulk water (96 atoms each), NumPy float64 on CPU, quippy threaded over the
frames; wall time per stage, measured outside the profiler:

| stage | dipole | polarisability |
|---|---|---|
| SOAP descriptors (external) | 1.91 s | 2.95 s + 3.90 s (6 Å scale) |
| neighbour pairs | 0.01 s | 0.01 s |
| bond basis V (3.4) | 0.02 s | 0.04 s |
| cross group (3.5) | 0.05 s | 0.04 s |
| 2b descriptor (3.7) | 0.02 s | — |
| 3b descriptor (3.7) | — | 0.44 s |
| 3b angular kernel W (3.8) | — | 2.54 s |
| kernel + site scalars + readout (3.9) | 0.01 s | 0.05 s |
| **total** | **2.10 s = 437 µs/atom** | **12.9 s = 2690 µs/atom** |

- The descriptor is 91 % of the dipole model and 53 % of the polarisability model. The part the
  port writes is ≈ 20 µs/atom (dipole) and ≈ 600 µs/atom (polarisability) in NumPy, almost all of
  the latter in the 3b kernel.
- The 3b kernel's time is interpreter overhead (one small matrix product per centre from a Python
  loop), not arithmetic. At r_c = 4.0 Å a water atom has 25 neighbours (15.5 at 3.5 Å): 2400 ordered
  pairs and 58 000 ordered triplets per frame. Building `W_i` is 58 000 × 144 × 6 ≈ 50 MFLOP per
  frame, the 3b descriptor ≈ 4 MFLOP; a compiled triplet loop does both in a few ms.
- Kernels and site scalars are small GEMMs: per polarisability frame ≈ 21 MFLOP for the two SOAP
  kernels (96 × 500 × 216 each), 1.7 MFLOP for the 3b kernel, ≈ 4 MFLOP for the site scalars. The
  dipole model is ≈ 12 MFLOP per frame in total. The non-descriptor pass is of order 1 MFLOP per
  atom (polarisability) and 0.15 MFLOP per atom (dipole); the port's floor is the descriptor.

---

## 7. Port checklist

1. Take `q_i` (and `q6_i` for polarisability) from soap_turbo as is.
2. One pair list per frame at r_c, both orders, centre-sorted; one Chebyshev recurrence at
   `max(C, C_x, C_s)` channels serves the bond basis, the cross group and the 2b/3b descriptors.
3. Triplets (polarisability only): all ordered neighbour pairs of each centre, ~600 per atom.
4. Segmented sums over the centre build `V_i`, the cross vectors, `W_i`, `q^{2b}_i`, `q^{3b}_i`
   with the column orders of section 4; the stored α assume them.
5. Per atom: squared dot products with the sparse points, the `[M_t] × [M_t, K_m]` products,
   the `D × K_m` contraction with `v_i` / `W_i`; accumulate over atoms.
6. Validate against `gap_dipole.predict` on a few frames: agreement to ~1e-10 relative once
   `q_i` agrees. Stage by stage: `V_i` (`data._neighbor_dipole_V`), `v_i` (`gap_dipole._v_atoms`),
   `W_i` (`gap_models.angular_cross_tensor`), `q^{3b}_i` (`gap_models.three_body_descriptor`),
   then the site scalars and the sum. The hand-rolled identities in `tests/test_gap_dipole.py`
   (`test_chebyshev_pair_identity`, `test_dipole_cross_group_identity_and_fit`,
   `test_pol_cross_group_identity_and_column_count`, `test_site_models_column_count_and_exact_recovery`,
   `test_unit_vector_geometry_used_by_fit_and_predict`) are the smallest executable statements
   of these conventions.
7. Model file: `model_io.save_gap_model(model, "<run>/gap_files/<name>.gap")` writes the
   TurboGAP layout, `model_io.load_gap_model` reads it back (round trip tested against
   `gap_dipole.predict`). The `.gap` file starts with a `tensor_readout` block (target, species
   order, `rcut`, `n_channels`, `radial`, `cutoff_function`, `species_pairs`, `unit_vector`,
   `cross_channels`, `cross_l_max`), then one block per site model and central species:
   `soap_turbo` (standard hypers), `site_2b` / `site_3b` (`n_channels`, `l_max`); the 6 Å scale
   is a second run of `soap_turbo` blocks. Every block carries `zeta`, `delta` (= δ_m),
   `n_columns` (= K_m), `desc_sparse` (`[M_t, Q_m]`, QUIP sparseX layout: one value per line,
   point-major) and `alphas_sparse` (`[M_t, K_m]`, one sparse point per line). Paths are
   relative to the directory above `gap_files/`.
