# Neumann–Kelvin wave resistance of a bare sailing-yacht canoe body

## Context

`ignaziomviola/wave-resistance` holds one commit: a Michell thin-ship solver with a sound numerical
kernel but a geometry layer that cannot ingest any real hull, and no CAD path at all. It is to be
deleted and replaced. The deliverable is a **defensible Neumann–Kelvin (NK) panel method** for the
linear potential-flow wave resistance of a bare canoe body, with Sysser 01 from the Delft Systematic
Yacht Hull Series as the worked case.

This plan replaces an earlier over-scoped version. Changes made in response to review: NK is the only
production method; Michell survives only as an analytic oracle; the DSYHS regression leaves the core;
the far-field spectrum is corrected to two branches; the NK integral equation must be complete and
verified before solver code; compute budgets are explicit; and the reference condition comes from
official Delft data. Claims retained only where derivation or data support them.

### One review point corrected by measurement

The review proposed establishing the benchmark by solving heave **and trim** to match the official
displacement and LCB. I tested that. It does not work, and the diagnosis matters.

At the official draught Tc = 127.040 mm with **zero trim**, my pipeline gives:

| | mine | official | diff |
|---|---|---|---|
| ∇c | 37.6008 dm³ | 37.6136 | **−0.034 %** |
| Aw | 55.8279 dm² | 55.8566 | **−0.051 %** |
| Lwl | 1607.99 mm | 1600.00 | +0.499 % |
| Bwl | 512.95 mm | 507.20 | +1.134 % |
| Sc | 65.5932 dm² | 64.2534 | +2.085 % |
| Cp | 0.562285 | 0.564414 | −0.377 % |
| Cwp | 0.676847 | 0.688296 | −1.663 % |
| LCB (from midship) | −34.333 mm | −36.640 | +2.31 mm |

Displacement and waterplane area agree to within 0.05 %. My earlier error was only in *inferring* the
draught from Lwl = 1.600 m instead of from displacement — the hydrostatics themselves reconcile.

Solving (heave, trim) to force ∇c and LCB to match exactly converges to a bow-up trim of −0.5704°
(waterplane z = 134.4246 − 9.9564×10⁻³·x mm) and hits ∇c to +0.010 % and LCB to 0.006 mm — but makes
every other quantity **worse**: Lwl +0.664 %, Bwl +1.285 %, Sc +2.291 %, Aw +0.444 %, Cwp −1.484 %.
So attitude does not explain the residual.

The official table is internally consistent (Cwp = Aw/(Lwl·Bwl) and Cm = Am/(Bwl·Tc) both check out on
its own numbers), so it describes a real hull — just not quite this surface. The likeliest cause is in
the file's own provenance: its header path reads *"Rhino modellen na inmeten 2012"* (Rhino models
after measuring, 2012) and the patch is named *"Gerebuild oppervlak"* (reconstructed surface). This is
a 2012 re-measurement of the physical model; the published hydrostatics may derive from the original
1981 lines. A 1–2 % difference in Bwl and Sc between an original lines plan and a re-measured hull is
unremarkable, and Sc — the quantity most sensitive to local surface fairness — shows the largest gap.

**Consequence for the plan:** reconciling this is a gated task against the downloaded official
dataset (§6), not something to assume away. The reference condition is set by **displacement at zero
trim**, with Lwl, Bwl, Sc, Cp, Cwp, LCB as reported validation outputs. Prescribed sinkage and trim
are offsets from it. Whether a trim is warranted is decided by the data, not by curve-fitting to LCB.

---

## 1. Precise mathematical scope

Steady, deep-water, inviscid, irrotational flow; linearised free-surface condition; **exact** body
boundary condition on the actual wetted hull at a **prescribed** attitude. Output: wave resistance
R_w of a bare canoe body. Nothing else is claimed.

Explicitly **not** modelled: viscous and form resistance; circulation, leeway, lift, appendages and
induced resistance (a source-only hull representation carries no circulation); nonlinear free-surface
effects; dynamic sinkage and trim as a solved equilibrium; breaking and spray; finite depth.

Geometry is retained as a full hull internally and no downstream code assumes port/starboard
symmetry, but only the **upright symmetric** case is validated in the first milestones. Heeled and
asymmetric resistance is gated behind verification of the two-branch spectrum (§7).

## 2. Complete NK formulation

**Frame.** Origin in the undisturbed free surface, x forward, z up, fluid in z < 0. Onset flow of
speed U in the −x direction, so upstream is x → +∞ (ahead of the bow) and the wake trails to
x → −∞. Total potential Φ = −Ux + φ. k₀ = g/U².

**Boundary-value problem.** With S_H the wetted hull, Γ = S_H ∩ {z = 0} the waterline, and **n** the
unit normal on S_H directed **into the fluid**:

1. ∇²φ = 0 in the fluid.
2. U²φ_xx + g φ_z = 0 on z = 0 outside the waterplane.
3. ∂φ/∂n = U n_x on S_H.
4. ∇φ → 0 as z → −∞; no waves upstream, φ → 0 as x → +∞. Enforced by the Rayleigh
   μ → 0⁺ prescription in the Green function.

**Green function.** G(P;Q) satisfying 1, 2, 4 for P ≠ Q, with Rankine part −1/(4π r) and image
term +1/(4π r₁), r₁ measured to (ξ, η, −ζ), plus the wave part. The wave part is implemented from a
primary reference (Noblesse 1981, *Alternative integral representations for the Green function of the
theory of ship wave resistance*; Newman 1987, *Evaluation of the wave-resistance Green function*)
rather than transcribed from memory. **Its constants are not asserted — they are fixed by the two
verification gates in §5 (V4, V5).**

**Representation and integral equation.** Source density σ on S_H:

  φ(P) = ∬_{S_H} σ(Q) G(P;Q) dS_Q

  ½σ(P) + ⨍∬_{S_H} σ(Q) ∂_{n(P)}G(P;Q) dS_Q + 𝒲[σ](P) = U n_x(P),   P ∈ S_H

where ⨍ is the principal value and 𝒲 is the **waterline term**: a line integral over Γ of the form
(U²/g)∮_Γ σ(Q) 𝒦(P;Q) n_x(Q) dl_Q, arising from Green's theorem applied to the part of z = 0 enclosed
by Γ, where the Kelvin G satisfies condition 2 but that plane is not a fluid boundary.

The sign of the jump term and the kernel 𝒦 and its coefficient are **derived explicitly in milestone
M1** before any solver code, and each is verified independently:

- the jump term and panel integration by the analytic sphere in unbounded flow (V3);
- 𝒲 by the free-surface residual test (V6) — with 𝒲 omitted or wrong, the residual of condition 2
  evaluated at off-body field points on z = 0 near Γ does not converge under refinement.

A waterplane-lid closure is the documented fallback if the line-integral form cannot be verified; the
difference between the two is then reported as a discretisation study, not hidden.

**Far-field amplitudes — both branches.** The free-wave field is a superposition over wave angle
θ ∈ (−π/2, π/2). Writing λ = sec θ and keeping **both** signs of the transverse wavenumber:

  A_±(λ) = C(λ) ∬_{S_H} σ(Q) exp(k₀λ²ζ) exp(−i k₀λ(ξ ± √(λ²−1) η)) dS_Q + (waterline contribution)

  R_w = (ρg²/πU²) ∫₁^∞ ½(|A₊(λ)|² + |A₋(λ)|²) λ²(λ²−1)^(−1/2) dλ

For a symmetric hull A₋ = A₊* and this collapses to the single-branch form, which is the form I have
already verified numerically (§5, V1–V2). The earlier plan's single-branch expression was therefore
the symmetric special case, as the review correctly identified; the ½ and the symmetric reduction
each get a unit test.

Resistance is taken from the far-field amplitude, not from hull-pressure integration, because it is
far less sensitive to panel-level pressure error. An independent pressure/energy evaluation is added
later as a consistency check (V11), not as the primary route.

**Michell as oracle only.** The verified thin-ship functional
a(λ) = ∬ Y_X e^(−λ²Z/Fn²) e^(−iλX/Fn²) dXdZ, I = ∫₀^∞ √(1+t²)|a|²dt, R_W = 4ρgL³I/(πFn²),
C_W = 8I/(πFn⁴(S/L²)) is retained **only** for the analytic Wigley hull, restricted to upright
symmetric thin hulls, to anchor the NK normalisation and the thin-body limit. There is no general
Michell solver and no Michell CLI option.

**Hogner** appears only as an internal closure σ = U n_x used to (a) exercise the far-field kernel
without the solve and (b) anchor the resistance normalisation against the Michell oracle in the thin
limit. It is not a fidelity level and is not advertised as one. That it is the zeroth iterate of the
integral equation above will be **derived for this exact convention, jump term and normal
orientation** in M1, or dropped; either way it tests normalisation only and is not a validation of the
Kelvin influence matrix.

No expected ordering between methods is asserted. Differences between models are reported as
model-discrepancy indicators, never as uncertainty.

## 3. Minimum module structure

`src/wave_resistance/`

| Module | Responsibility |
|---|---|
| `iges.py` | IGES reader, **Type 128 only** for M0. Sections by column 73, delimiters and units from the Global record, `D`→`E` exponents, Hollerith strings, payload grouped by DE pointer (cols 65–72). Unsupported entities skipped with a report. Output in metres. |
| `nurbs.py` | Cox–de Boor basis and derivatives; surface point, first derivatives, normal; numpy-vectorised. |
| `hull.py` | Full-hull geometry, explicit mirroring recorded as provenance, no downstream symmetry assumption. Two meshes: a **fine** mesh for geometry/hydrostatics and a **coarse** mesh for NK panels. `Attitude(sinkage, trim, heel)` as prescribed offsets from the reference condition. Clipping at z = 0, watertight check, signed volume, outward normals. |
| `hydrostatics.py` | Divergence-theorem integrals over the full closed wetted surface: ∇c, LCB, VCB, Aw, LCF, Sc, Am, Cp, Cm, Cb, Cwp. Reference-condition solver: heave for target displacement (trim optional, off by default). |
| `wigley.py` | Analytic Wigley hull and its **closed-form** Michell amplitude — oracle only. |
| `greens.py` | Kelvin Green function and its gradient; Rankine + image + wave part; singular and near-singular panel quadrature. |
| `panels.py` | Flat quad/tri panels; exact Hess–Smith Rankine influence coefficients; **full-kernel panel integration** of the wave part (never centroid × area) with order set by k₀ × panel size and local phase. |
| `nk.py` | Assembly, dense solve, σ, far-field A₊/A₋, R_w; residual and conditioning diagnostics. |
| `spectrum.py` | λ = √(1+t²) substitution, phase-locked panels, nested Gauss, tail certification. Consumes any A_±(λ). |
| `cli.py` | `wave-resistance hydrostatics` and `wave-resistance nk`. |

`dsyhs.py` is an **optional comparison module**, outside the solver path, with coefficients in an
editable data file. Deferred: IGES 126/142/144, plotting, report generation.

## 4. Staged milestones

- **M0 — geometry.** Repo reset, packaging, `iges.py` (128 only), `nurbs.py`, `hull.py`,
  `hydrostatics.py`. Gate: V1, V7, V8, V9.
- **M1 — formulation, no solver.** Written derivation of the jump term, 𝒲, the source normalisation
  and A_±; `greens.py` prototype; measured kernel throughput. Gate: V4, V5, V6, plus a documented
  derivation of (or the dropping of) the Hogner-as-zeroth-iterate claim.
- **M2 — Rankine solver.** `panels.py` + solve with the free-surface part switched off. Gate: V3.
- **M3 — full NK.** Assembly, solve, far-field resistance at **a few Delft measurement speeds,
  Fn ≥ 0.20**. Gate: V2, V10, V11, V12.
- **M4 — Sysser 01 validation.** §6. Gate: quantitative reconciliation, or a written statement of the
  unresolved discrepancy.
- **M5 — deferred.** Two-branch heeled/asymmetric resistance; `dsyhs.py`; wider Fn sweep; plots.

Each milestone commits to `claude/wave-resistance-sailing-boats-vh4pgy`. No PR unless asked.

## 5. Verification, with acceptance thresholds

Numerical verification, separate from experimental validation.

| | Test | Threshold |
|---|---|---|
| V1 | IGES/NURBS reference points, derivatives by finite difference, bounding box, partition of unity, endpoint interpolation | 1e−10 relative |
| V2 | Analytic Wigley Michell spectrum and resistance; NK thin-body limit on progressively thinner Wigley hulls (B/L = 0.05, 0.02, 0.01) | oracle reproduced to 1e−6; NK→Michell within discretisation error, trending as B/L² |
| V3 | Sphere in unbounded uniform stream: σ = (3/2)U n_x. Fixes jump-term sign **and** panel integration | 0.5 % at 1200 panels, second-order convergence |
| V4 | Green function against independently published values / limiting cases (large depth, large separation → Rankine + image) | 1e−6 relative |
| V5 | Green function derivatives by finite difference; free-surface condition U²G_xx + gG_z = 0 tested pointwise on z = 0 | 1e−6 relative |
| V6 | Free-surface residual at **off-collocation** field points on z = 0 near Γ, with and without 𝒲 | residual → 0 under refinement with 𝒲; fails without |
| V7 | Analytic hydrostatics: box, half-cylinder, Wigley (Cb = 4/9, Cwp = Cm = Cp = 2/3) | 1e−8 |
| V8 | Outward normals, mirror orientation, watertight clipping, signed volume vs independent section-area integration | 1e−9 |
| V9 | Mesh convergence of ∇c, Sc, Aw | 5 significant figures |
| V10 | NK convergence under panel **and** quadrature refinement, independently | R_w to 1 % between successive levels |
| V11 | Far-field R_w vs independent pressure/energy evaluation | 5 % |
| V12 | Body-condition residual at off-collocation points; global flux conservation; matrix condition number reported every run | residual < 1 % of U; flux < 1e−6·U·S |
| V13 | Submerged-body wave resistance against an independently sourced oracle | 5 %, **only if** such an oracle is actually located; otherwise recorded as not done |

V3 and V6 are the decisive gates. "NK zeroth iterate → Hogner" is recorded as a normalisation check
only. The earlier plan's heel test — volume conserved under rotation at fixed sinkage — was wrong and
is removed; rotation changes immersed volume, so volume is conserved only after solving the required
heave/trim equilibrium, which is out of scope here.

## 6. Sysser 01 validation against official data

Download and use the primary Delft releases rather than inferred values:

- geometry — figshare 21501330
- hydrostatics — 4TU 21501375
- model- and full-scale measurements — 4TU 21501402

Tasks:

1. **Reconcile hydrostatics.** Reproduce the official zero-heel model-scale table. Reference
   condition set by matching official ∇c = 0.0376136 m³ at zero trim; Lwl, Bwl, Sc, Aw, Cp, Cm, Cwp,
   LCB reported as outputs with signed differences. Starting point is the table in the Context section
   above (∇c and Aw already within 0.05 %). Investigate datum, loading and static-trim conventions,
   and whether the published table describes the original lines rather than the 2012 reconstruction.
   If the Lwl/Bwl/Sc gap cannot be closed, state it as an unresolved geometry-provenance discrepancy
   with its magnitude, and carry it as a bias term in every downstream comparison.
2. **Resistance comparison at measured attitude.** Take total resistance and the recorded dynamic
   sinkage and trim per speed from the measurement workbook; run NK at those prescribed attitudes.
   Compare, stating both explicitly:
   - predicted R_w + R_f against measured total resistance, with the k = 0 assumption named; and
   - predicted R_w against experimentally derived residuary resistance, **labelled a proxy, not
     ground truth** — the Delft residuary resistance follows a friction subtraction with form factor
     effectively zero and therefore contains wave, nonlinear and some viscous-form resistance.
3. Report experimental and numerical tolerances, and separate discretisation error from model-form
   discrepancy.

The 1998 regression is **not** a validation target. If shipped, it is an optional comparison module,
transcribed from a reliable source and tested against that source's published conventions, with
low-Fn negative values preserved and warned about, never clamped. For the record: the review's
statement that a7 multiplies (LCB_fpp/Lwl)² and that LCB_fpp/LCF_fpp is the separate a6 term agrees
with what I inferred numerically — my inference rested on a physical impossibility (Cr,c reaching
−0.95, i.e. negative resistance approaching displacement weight) rather than on plausibility — but the
code will still transcribe from a citable source rather than rely on that inference.

## 7. Compute budget and feasibility

Dense assembly is O(N²) kernel evaluations and O(N³) solve **per Froude number**, so:

- Fine geometry mesh and coarse NK mesh are separate objects. NK target **800–2000 panels**.
- At N = 1500: complex influence matrix 1500² × 16 B = 36 MB; LU ≈ 1.1 GFLOP, well under a second.
  The cost is 2.25×10⁶ panel-pair kernel evaluations per speed.
- **Hard budget per speed: ≤ 8 GB peak memory, ≤ 10 min assembly, ≤ 1 min solve.** Exceeding it fails
  the milestone rather than silently running.
- M1 measures kernel throughput **before** any promise about a speed sweep. M3 runs a handful of
  Delft measurement speeds at Fn ≥ 0.20. No 0.10–0.60 sweep is promised.
- Deterministic tabulation/interpolation of the Green function is considered **only after** pointwise
  kernel accuracy passes V4 and V5.

## 8. Explicitly deferred

General Michell solver and its CLI option; Hogner as an advertised fidelity level; the DSYHS
regression in the core solver; IGES types 126, 142, 144; plots and report generation; heeled and
asymmetric resistance until the two-branch spectrum passes verification; free sinkage and trim
equilibrium; finite depth; appendages, lift, leeway and induced resistance; wave-pattern
visualisation; the full 51-speed sweep.

## End-to-end check

```
pip install -e ".[test]"
pytest -q                                   # V1-V13 with the thresholds above
wave-resistance hydrostatics data/SYSSER01_surface.igs --displacement 0.0376136
      # reports every hydrostatic with its signed difference from the official table
wave-resistance nk data/SYSSER01_surface.igs --displacement 0.0376136 \
      --fn 0.25,0.30,0.35,0.40 --panels 1200
      # reports R_w, body-condition residual at off-collocation points, flux error,
      # condition number, and panel/quadrature convergence
```
