# RSM-WSGG: A Response-Surface Weighted-Sum-of-Gray-Gases Model for H₂O/NH₃ Gas Radiation

This repository contains the **data, code and figure-reproduction scripts** accompanying the
manuscript *"A response-surface weighted-sum-of-gray-gases model for H₂O/NH₃ gas radiation"*.

It provides, in a single public archive:

1. the **line-by-line (LBL) reference calculations** — both the emissivity database used for
   training/validation and the LBL reference solutions of the one-dimensional radiative
   transfer equation (RTE) for every test case;
2. the **validation data for Cases 1.1–3.2** (prescribed temperature and species profiles,
   together with their LBL reference solutions), plus the two additional realistic-flame
   cases (Cases 4.1–4.2);
3. the **figure- and table-reproduction scripts** that regenerate every figure and table in
   the manuscript;
4. this **English README**, together with the Chinese `README.md`.

---

## 1. Repository structure

```
RSM_WSGG/
├── README.md                       # Chinese README
├── README_en.md                    # this file (English README)
│
├── config/
│   ├── path.py                     # path configuration (edit before first use)
│   └── coefficients.json           # default coefficient-set selection
│
├── notebooks/
│   ├── gasmix_emiss_calc.ipynb     # LBL emissivity database generation (requires AbscDB)
│   ├── model_training.ipynb        # RSM-WSGG training and WSGG coefficient generation
│   ├── validation.ipynb            # emissivity and 1D RTE validation
│   ├── validation_publication_plots.ipynb  # journal-style figure/table reproduction
│   ├── ng_benchmark.ipynb          # N_g = 4–7 accuracy and RTE-cost benchmark
│   ├── spectral_convergence.ipynb  # spectral-resolution convergence study
│   └── separator.ipynb             # utility
│
├── src/
│   ├── gas_rte_solver.py           # RTE solver (solve_rte, solve_wsgg_dom, getLBLsolution,
│   │                               #             get_kappas)
│   ├── get_emis_abso.py            # LBL absorption-coefficient / emissivity interface
│   ├── data_preparer.py            # training / validation dataset preparation
│   ├── profile_generator.py        # synthetic T, Y_H2O, Y_NH3 profiles for Cases 1.1–3.2
│   ├── generate_validation_plots.py# all publication figures and tables
│   ├── ExtractFortranDB.py         # spectral-database extraction
│   ├── utils.py                    # I/O helpers (read_from_excel, resolve_lbl_reference, …)
│   └── models/
│       ├── wsgg_model.py           # WSGGModel (beta, gamma) container and forward evaluation
│       ├── rsm_wsgg.py             # RSM-WSGG fitting
│       ├── wang_wsgg.py            # Wang et al. WSGG model (comparison model)
│       └── wsgg.py                 # generic WSGG helpers
│
├── input/
│   ├── MixEmiss_p1.0_X0.2_Mr0.1-4.0_40_T40_path40.npy  # main emissivity dataset
│   ├── MixEmiss_p1.0_X0.2_Mr0.1-4.0_15_T22_path22.npy  # independent validation dataset
│   ├── MixEmiss_*.npy                                   # additional datasets
│   ├── coefficients_from_wang.xlsx                      # Wang et al. coefficients
│   └── profiles/
│       ├── Case1.1_L1.0m.npy … Case3.2_L1.0m.npy        # Cases 1.1–3.2 profiles
│       ├── Case4.1_L1.0m.npy                            # realistic flame, L = 1.0 m
│       └── Case4.2_L5.0m.npy                            # realistic flame, L = 5.0 m
│
└── results/
    ├── coefficients_<timestamp>_Ng<N>_Exp4_Tref1000_bs<B>.xlsx   # fitted coefficients
    ├── coefficients_<…>.log                                       # training log
    ├── lbl/
    │   └── Case1.1_L1.0m.npy … Case4.2_L5.0m.npy       # LBL reference RTE solutions
    ├── publication_notebook/                            # figures + tables (PNG/PDF/TeX)
    ├── ng_benchmark/                                    # N_g benchmark (csv/tex/png)
    ├── spectral_convergence/                            # convergence study (csv/png)
    └── img/                                             # exploratory figures
```

---

## 2. Environment setup

There is **no one-click setup script in the repository**; install the dependencies directly:

```bash
pip install numpy scipy pandas matplotlib openpyxl numba tqdm
# torch is only needed for model training (notebooks/model_training.ipynb)
```

Before the first run, edit `config/path.py` so that the absolute paths point to your local
`AbscDB` spectral database and working directories. The LBL *emissivity database* shipped in
`input/` is self-contained, but regenerating it from scratch (`gasmix_emiss_calc.ipynb`) or
recomputing the LBL reference solutions requires `AbscDB`.

---

## 3. Reproducing the figures and tables

All manuscript figures and tables are produced by
`src/generate_validation_plots.py` (batch mode) or interactively by
`notebooks/validation_publication_plots.ipynb`.

```bash
cd src
python generate_validation_plots.py --groups all \
       --coefficients ../results/coefficients_20260326_122927_Ng6_Exp4_Tref1000_bs16384.xlsx
```

`--groups` accepts `parity`, `emissivity`, `cases` (or `all`). Outputs are written to
`results/publication_notebook/` as PNG (600 dpi), PDF and, for tables, LaTeX/CSV.

### 3.1 Mapping between manuscript items and output files

| manuscript item | output file(s) |
|---|---|
| **Fig. 2** — parity of total emissivity | `parity_models_vs_lbl.png` (and `parity_rsm_kde.png`) |
| **Fig. 3** — emissivity vs. temperature | `emissivity_vs_temperature_mr_0p500.png`, `…_mr_3p500.png` |
| **Fig. 4** — emissivity vs. pressure path length (varying $M_r$) | `emissivity_vs_path_length_mr_0p500.png`, `…_mr_3p500.png` |
| **Fig. 5** — emissivity vs. pressure path length (varying $T$) | `emissivity_vs_path_length_t_1100K.png`, `…_t_1700K.png` |
| **Fig. 6** — isothermal homogeneous cases | `case_validation_Case1.1/1.2/1.3_L1.0m.png` |
| **Fig. 7** — non-isothermal homogeneous cases | `case_validation_Case2.1/2.2_L1.0m.png` |
| **Fig. 8** — non-isothermal inhomogeneous cases | `case_validation_Case3.1/3.2_L1.0m.png` |
| **Fig. 9** — realistic-flame cases (new) | `case_validation_Case4.1_L1.0m.png`, `case_validation_Case4.2_L5.0m.png` |
| **Table 3** — error statistics | `case_error_summary.tex` / `.csv` / `.png` |

> **Note on figure numbering.** In the previous version of the manuscript the
> error-statistics summary was numbered **Fig. 9**; it is now **Table 3**. The newly added
> realistic-flame figure takes the number **Fig. 9**. Earlier references in the response
> letter to *"Figs. 2–9"* therefore denoted "all figures *and* the summary table" and should
> now be read as **"Figs. 2–9 and Table 3"**. Please also consult the manuscript itself for
> the authoritative numbering of the final typeset version.

### 3.2 Validation cases and their data

Each `input/profiles/CaseX.Y_LZ.Zm.npy` is a structured array with fields
`x` (m), `Y_H2O` (–), `Y_NH3` (–), `M_ratio` (–), `T` (K).
The matching LBL reference is `results/lbl/CaseX.Y_LZ.Zm.npy`, an `(N, 3)` array with
columns `[x, q, -dq]`, where `q` is the radiative heat flux and `-dq` the radiative source
term.

| case | file | $L$ (m) | grid | description |
|---|---|---|---|---|
| 1.1–1.3 | `Case1.1…1.3_L1.0m.npy` | 1.0 | 100 | isothermal, homogeneous |
| 2.1–2.2 | `Case2.1…2.2_L1.0m.npy` | 1.0 | 100 | non-isothermal, homogeneous |
| 3.1–3.2 | `Case3.1…3.2_L1.0m.npy` | 1.0 | 100 | non-isothermal, inhomogeneous |
| 4.1 | `Case4.1_L1.0m.npy` | 1.0 | 81 | realistic flame (see below) |
| 4.2 | `Case4.2_L5.0m.npy` | 5.0 | 81 | same flame, longer path |

Cases 4.1 and 4.2 use temperature and species fields taken directly from the published
one-dimensional solution of a **laminar premixed NH₃/H₂/N₂–air counterflow flame**
(Richter, Barlow, Dawson, Dreizler and Geyer, *Proc. Combust. Inst.* **42** (2026),
TU-Datalib, DOI [10.48328/tudatalib-2271](https://doi.org/10.48328/tudatalib-2271),
CC BY 4.0). No chemical kinetics are solved. The retained portion is the contiguous
flame-front region in which the molar ratio stays inside the training envelope
$0.1 \le M_r \le 4.0$ ($T$ from 423 to 1471 K); the fully burnt region is excluded because
NH₃ is consumed there and $M_r \to \infty$.

### 3.3 A note on the comparison model (Wang et al.)

The Wang et al. model is stated for 300–2300 K and pressure path lengths of 0.1–20 m. In
the figures, data outside this envelope are marked explicitly:

* in the parity plot, out-of-envelope Wang points are drawn as grey hollow squares and
  their statistics are reported in a separate box, `Wang (extrapolated)`;
* in the emissivity-vs-temperature figures, the curves with $P_tL > 20$ bar·m are drawn
  dashed (caption: "outside Wang range");
* in the emissivity-vs-path-length figures, the region $P_tL > 20$ bar·m is shaded grey.

Only the unshaded / non-dashed region should be used for a strict quantitative comparison.
The two models also rest on different spectroscopic databases (HITEMP2010/HITRAN2020 here
versus HITEMP2016/HITRAN2010 in Wang et al.), which further limits strict comparability.

---

## 4. Data availability

| item | location |
|---|---|
| LBL reference emissivity database | `input/MixEmiss_*.npy` |
| LBL reference RTE solutions (all 9 cases) | `results/lbl/Case*.npy` |
| Input profiles, Cases 1.1–3.2 | `input/profiles/Case1*.npy` … `Case3*.npy` |
| Input profiles, realistic-flame Cases 4.1–4.2 | `input/profiles/Case4*.npy` |
| Fitted RSM-WSGG coefficients | `results/coefficients_*.xlsx` |
| Comparison-model coefficients | `input/coefficients_from_wang.xlsx` |
| Figure / table reproduction scripts | `src/generate_validation_plots.py`, `notebooks/validation_publication_plots.ipynb` |
| Supporting studies (N_g, spectral convergence) | `notebooks/ng_benchmark.ipynb`, `notebooks/spectral_convergence.ipynb` |
| Reproduced figures and tables | `results/publication_notebook/` |

The coefficient set used in the manuscript is

```
results/coefficients_20260326_122927_Ng6_Exp4_Tref1000_bs16384.xlsx
```

| sheet | shape | meaning |
|---|---|---|
| `beta`  | `(N_g+1, 15)` | grey-gas weight logits (softmax); first row = transparent gas |
| `gamma` | `(N_g+1, 5)`  | absorption-coefficient polynomial (softplus); first row $\kappa = 0$ |

Model constants: $N_g = 6$ absorbing gray gases + 1 transparent gas, polynomial order
`Exp = 4`, $T_\text{ref} = 1000$ K, $X = X_{H_2O} + X_{NH_3} = 0.2$ (fixed),
$P_t = 1$ bar.

---

## 5. Citation

If you use this code or data, please cite the original manuscript and this repository:

```
https://github.com/huatyrz/RSM_WSGG
```

The realistic-flame profiles in Cases 4.1–4.2 are derived from the dataset of
Richter et al. (TU-Datalib, DOI 10.48328/tudatalib-2271, CC BY 4.0); please cite that
dataset as well when using those cases.
