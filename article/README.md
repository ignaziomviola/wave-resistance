# Journal-style displacement-yacht test case

This directory contains an Overleaf-ready JFM-style article following the
VOILAb paper-writing guide supplied in the linked Overleaf project.

## Reproduce the numerical results

From the repository root:

```bash
python3 examples/journal_test_case.py
```

The script regenerates:

- `data/resistance_curve.csv`: all 13 Froude-number cases and three hull grids;
- `data/wave_patterns.npz`: regular wave-pattern grids at six Froude numbers;
- `data/test_case_summary.json`: geometry, hydrostatics and numerical diagnostics;
- all PDF figures in `figures/`.

## Compile the article

```bash
cd article
latexmk -pdf main.tex
```

The final manuscript is `main.pdf`. The directory includes the `jfm.cls` and
`jfm.bst` files from the supplied VOILAb Overleaf template, so the complete
`article/` directory can be uploaded to Overleaf without additional files.

## Scope

The resistance coefficient is obtained from the three-dimensional linear
Havelock source distribution and Kochin far-field integral. Wave-pattern plots
are independently normalised phase reconstructions from the same spectrum;
they are not dimensional near-field elevations. The test is suitable for
method verification and hull-form screening, not certification of a specific
yacht.
