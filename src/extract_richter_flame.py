#!/usr/bin/env python3
"""Case 5: 1-D RTE on prescribed profiles taken from a published NH3/H2/N2-air counterflow flame.

Reviewer #3, Comment 7 -- a 1-D LBL solve on *prescribed* temperature and species
profiles taken from a published NH3 or NH3/H2 flame is computationally equivalent
to Cases 2.1-3.2 and is what the reviewer asked for.  This script extracts such
profiles from an openly licensed dataset and writes them in the same
``Case*.npy`` format used by ``input/profiles/``.

Source dataset
--------------
M. Richter, R.S. Barlow, J.R. Dawson, A. Dreizler, D. Geyer,
"High-Temperature NO-LIF Measurements in Laminar Premixed NH3/H2/N2-Air
Counterflow Flames", Proceedings of the Combustion Institute 42.
TU-Datalib, DOI 10.48328/tudatalib-2271, CC BY 4.0.
File: ``Richter_2026_NO-LIF_NH3-H2-N2_air_counterflow_publication_data_v2.h5``

The HDF5 archive holds 39 operating conditions (case_01 .. case_39) spanning

    phi        in {0.80, 1.10}
    strain Kg  in {400, 500, 600, 700, 800} 1/s
    X_H2_fuel  in {27, 33, 39, 45} vol-%  (X_N2_fuel = X_H2/3, X_NH3 = remainder)

and, for each condition, three independent 1-D chemical-kinetic solutions
(mechanisms ``STA`` [Stagni 2023], ``NAK`` [Nakamura 2017], ``HAN`` [Han 2020])
under ``/cases/case_XX/simulation/<mechanism>/``.  Each mechanism group provides

    x              flame-fixed coordinate            [mm]
    T              temperature                       [K]
    major/H2O      H2O  mole fraction                [-]
    major/NH3      NH3  mole fraction                [-]
    major/{H2,N2,O2}
    minor/{H,NH,NH2,O,OH}

which is exactly the (T, X_H2O, X_NH3) triple needed here.

Why only the flame-front slice is used
--------------------------------------
The model was trained for 0.1 <= M_r = X_H2O/X_NH3 <= 4.0 with X = X_H2O+X_NH3
= 0.2.  In a *burnt* ammonia flame NH3 is fully consumed, so M_r -> infinity on
the product side.  Feeding M_r = 1e6 into the fitted polynomial makes the
absorption coefficients diverge (verified: kappa ~ 1e30 at M_r = 1e8), which
would produce meaningless numbers rather than a validation.

The script therefore:

1. scans every (condition, mechanism) pair,
2. locates the longest *contiguous* run of grid points with 0.1 <= M_r <= 4.0,
3. keeps that slice (a genuine piece of the published flame: it straddles the
   preheat / reaction zone and carries a strong temperature gradient),
4. maps it affinely onto the normalised coordinate x/L of the 1-D RTE domain,
5. reports both the in-envelope statistics and the discarded fraction.

X is *not* rescaled by default: within the retained slice it stays close to the
training value 0.2 (typically 0.15-0.24), and X enters the model only through
the optical path p*X*L.  ``--rescale-x`` keeps M_r and renormalises
(X_H2O, X_NH3) so that their sum is exactly 0.2 (sensitivity check).

Usage
-----
    # 1) which condition/mechanism is best suited?
    python src/extract_richter_flame.py --scan --h5 ../Richter_2026_v2.h5 --top 12

    # 2) build the profile(s)
    python src/extract_richter_flame.py --make-profile \
        --h5 ../Richter_2026_v2.h5 --case 10 --mechanism STA --length 1.0

    # 3) (optional) same at a larger optical thickness
    python src/extract_richter_flame.py --make-profile \
        --h5 ../Richter_2026_v2.h5 --case 10 --mechanism STA --length 5.0 --suffix 2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Training envelope of the present RSM-WSGG model (Section 2.2 of the manuscript).
MR_MIN, MR_MAX = 0.1, 4.0
T_MIN, T_MAX = 300.0, 2400.0
X_TRAIN = 0.2

MECHANISMS = ("STA", "NAK", "HAN")


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def open_h5(path: Path):
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"h5py is required to read the dataset ({exc}); pip install h5py")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return h5py.File(path, "r")


def conditions_table(root) -> Dict[str, np.ndarray]:
    grp = root["conditions"]
    return {key: np.asarray(grp[key][0]) for key in grp.keys()}


def mechanism_profile(root, case_id: int, mechanism: str):
    grp = root["cases"][f"case_{case_id:02d}"]["simulation"][mechanism]
    x = np.asarray(grp["x"][0], dtype=float)
    T = np.asarray(grp["T"][0], dtype=float)
    h2o = np.asarray(grp["major"]["H2O"][0], dtype=float)
    nh3 = np.asarray(grp["major"]["NH3"][0], dtype=float)
    order = np.argsort(x)
    return x[order], T[order], h2o[order], nh3[order]


def mole_ratio(h2o: np.ndarray, nh3: np.ndarray) -> np.ndarray:
    return np.where(nh3 > 1e-12, h2o / np.where(nh3 > 1e-12, nh3, 1.0), np.inf)


def longest_run(mask: np.ndarray) -> Tuple[int, int]:
    """Return the (start, stop) of the longest contiguous True run."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0, 0
    breaks = np.flatnonzero(np.diff(idx) > 1)
    segments = np.split(idx, breaks + 1)
    best = max(segments, key=len)
    return int(best[0]), int(best[-1]) + 1


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #
def scan(root, mechanism: str, mr_min: float, mr_max: float) -> List[Dict]:
    cond = conditions_table(root)
    rows: List[Dict] = []
    n_cases = root["cases"].__len__()
    for case_id in range(1, n_cases + 1):
        try:
            x, T, h2o, nh3 = mechanism_profile(root, case_id, mechanism)
        except KeyError:
            continue
        X = h2o + nh3
        mr = mole_ratio(h2o, nh3)

        in_mr = (mr >= mr_min) & (mr <= mr_max)
        in_T = (T >= T_MIN) & (T <= T_MAX)
        usable = in_mr & in_T
        start, stop = longest_run(usable)

        # radiation weight ~ T^4 * X  (where the gas actually emits)
        weight = np.power(np.clip(T, 0.0, None), 4.0) * X
        weight_total = float(np.trapezoid(weight, x)) if x.size > 1 else 0.0

        row = {
            "case": case_id,
            "mechanism": mechanism,
            "phi": float(cond["phi"][case_id - 1]),
            "X_H2_fuel_pct": float(cond["H2_fuel"][case_id - 1]),
            "Kg": float(cond["Kg"][case_id - 1]),
            "n_pts": int(x.size),
            "n_in_envelope": int(usable.sum()),
            "n_longest_run": int(stop - start),
        }
        if stop > start:
            sl = slice(start, stop)
            seg_x, seg_T, seg_X, seg_mr = x[sl], T[sl], X[sl], mr[sl]
            seg_w = np.power(np.clip(seg_T, 0.0, None), 4.0) * seg_X
            w_seg = float(np.trapezoid(seg_w, seg_x)) if seg_x.size > 1 else 0.0
            row.update({
                "span_mm": float(seg_x.max() - seg_x.min()),
                "T_min": float(seg_T.min()),
                "T_max": float(seg_T.max()),
                "X_min": float(seg_X.min()),
                "X_max": float(seg_X.max()),
                "Mr_min": float(np.min(seg_mr[np.isfinite(seg_mr)])),
                "Mr_max": float(np.max(seg_mr[np.isfinite(seg_mr)])),
                "rad_weight_pct": 100.0 * w_seg / weight_total if weight_total > 0 else 0.0,
                "_start": start,
                "_stop": stop,
            })
        else:
            row.update({k: float("nan") for k in
                        ("span_mm", "T_min", "T_max", "X_min", "X_max",
                         "Mr_min", "Mr_max", "rad_weight_pct")})
            row["_start"] = row["_stop"] = 0
        rows.append(row)
    return rows


def print_scan(rows: Sequence[Dict], top: int) -> None:
    cols = ["case", "phi", "X_H2_fuel_pct", "Kg", "n_pts", "n_longest_run",
            "span_mm", "T_min", "T_max", "X_min", "X_max", "Mr_min", "Mr_max",
            "rad_weight_pct"]
    header = f"{'case':>4} {'phi':>5} {'H2%':>5} {'Kg':>5} {'pts':>5} {'run':>5} " \
             f"{'span':>7} {'Tmin':>6} {'Tmax':>6} {'Xmin':>7} {'Xmax':>7} " \
             f"{'Mrmin':>7} {'Mrmax':>7} {'radwt%':>8}"
    print(header)
    print("-" * len(header))
    ordered = sorted(rows, key=lambda r: (r["n_longest_run"], r["rad_weight_pct"]),
                     reverse=True)
    for row in ordered[:top]:
        print(f"{row['case']:>4} {row['phi']:>5.2f} {row['X_H2_fuel_pct']:>5.0f} "
              f"{row['Kg']:>5.0f} {row['n_pts']:>5} {row['n_longest_run']:>5} "
              f"{row['span_mm']:>7.3f} {row['T_min']:>6.0f} {row['T_max']:>6.0f} "
              f"{row['X_min']:>7.3f} {row['X_max']:>7.3f} "
              f"{row['Mr_min']:>7.3f} {row['Mr_max']:>7.3f} "
              f"{row['rad_weight_pct']:>8.2f}")
    print(f"\n(mechanism = {rows[0]['mechanism'] if rows else '-'}; "
          f"{len(rows)} conditions; showing top {top})")


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def build_profile(root, case_id: int, mechanism: str, length: float,
                  n_points: int, rescale_x: bool) -> Tuple[np.ndarray, Dict]:
    x_mm, T, h2o, nh3 = mechanism_profile(root, case_id, mechanism)
    X = h2o + nh3
    mr = mole_ratio(h2o, nh3)

    usable = (mr >= MR_MIN) & (mr <= MR_MAX) & (T >= T_MIN) & (T <= T_MAX)
    start, stop = longest_run(usable)
    if stop - start < 4:
        raise SystemExit(
            f"case_{case_id:02d}/{mechanism}: only {stop - start} usable points "
            f"inside 0.1 <= M_r <= 4.0 -- pick another condition (run --scan)."
        )
    sl = slice(start, stop)
    seg_x, seg_T, seg_h2o, seg_nh3 = x_mm[sl], T[sl], h2o[sl], nh3[sl]

    # affine map: flame slice  ->  normalised coordinate s in [0, 1]
    s_src = (seg_x - seg_x.min()) / (seg_x.max() - seg_x.min())
    s_new = np.linspace(0.0, 1.0, n_points)

    T_new = np.interp(s_new, s_src, seg_T)
    h2o_new = np.interp(s_new, s_src, seg_h2o)
    nh3_new = np.interp(s_new, s_src, seg_nh3)

    if rescale_x:
        total = h2o_new + nh3_new
        scale = X_TRAIN / np.where(total > 1e-12, total, 1.0)
        h2o_new = h2o_new * scale
        nh3_new = nh3_new * scale

    mr_new = mole_ratio(h2o_new, nh3_new)

    profile = np.zeros(n_points, dtype=[("x", "f8"), ("Y_H2O", "f8"),
                                        ("Y_NH3", "f8"), ("M_ratio", "f8"),
                                        ("T", "f8")])
    profile["x"] = s_new * length
    profile["Y_H2O"] = h2o_new
    profile["Y_NH3"] = nh3_new
    profile["M_ratio"] = mr_new
    profile["T"] = T_new

    cond = conditions_table(root)
    info = {
        "case": case_id,
        "mechanism": mechanism,
        "phi": float(cond["phi"][case_id - 1]),
        "X_H2_fuel_pct": float(cond["H2_fuel"][case_id - 1]),
        "X_NH3_fuel_pct": float(cond["NH3_fuel"][case_id - 1]),
        "Kg": float(cond["Kg"][case_id - 1]),
        "length_m": length,
        "n_points": n_points,
        "source_mm": f"{seg_x.min():.3f} .. {seg_x.max():.3f}",
        "flame_pts_used": int(stop - start),
        "flame_pts_total": int(x_mm.size),
        "T_min": float(T_new.min()),
        "T_max": float(T_new.max()),
        "X_min": float((h2o_new + nh3_new).min()),
        "X_max": float((h2o_new + nh3_new).max()),
        "Mr_min": float(np.min(mr_new[np.isfinite(mr_new)])),
        "Mr_max": float(np.max(mr_new[np.isfinite(mr_new)])),
        "pXL_max_bar_m": float(1.0 * (h2o_new + nh3_new).max() * length),
        "rescaled_x": bool(rescale_x),
    }
    return profile, info


def save_profile(profile: np.ndarray, info: Dict, profiles_dir: Path,
                 reports_dir: Path, suffix: int) -> Path:
    stem = f"Case5.{max(1, suffix)}_L{info['length_m']:.1f}m"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    npy_path = profiles_dir / f"{stem}.npy"
    np.save(npy_path, profile)

    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "Y_H2O", "Y_NH3", "M_ratio", "T_K"])
        for row in profile:
            writer.writerow([f"{row['x']:.6f}", f"{row['Y_H2O']:.8f}",
                             f"{row['Y_NH3']:.8f}", f"{row['M_ratio']:.6f}",
                             f"{row['T']:.4f}"])
    print(f"Saved {npy_path}")
    print(f"Saved {csv_path}")
    return npy_path


def draw_profile(profile: np.ndarray, info: Dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[skip] figure not produced ({exc})")
        return

    x_rel = profile["x"] / info["length_m"]
    X = profile["Y_H2O"] + profile["Y_NH3"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(x_rel, profile["T"], color="#C62828")
    axes[0].set_ylabel(r"$T$ (K)")
    axes[1].plot(x_rel, X, color="#1565C0", label=r"$X$")
    axes[1].plot(x_rel, profile["Y_H2O"], "--", color="#1565C0", alpha=0.6, label=r"$X_{H_2O}$")
    axes[1].plot(x_rel, profile["Y_NH3"], ":", color="#1565C0", alpha=0.6, label=r"$X_{NH_3}$")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].axhline(X_TRAIN, color="k", linewidth=0.7, alpha=0.4)
    axes[2].plot(x_rel, profile["M_ratio"], color="#00695C")
    axes[2].axhline(MR_MAX, color="k", linewidth=0.7, alpha=0.4)
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$M_r$")
    for ax, label in zip(axes, ("Temperature", "Mole fractions", "Mole ratio")):
        ax.set_xlabel(r"$x/L$")
        ax.set_title(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", type=Path,
                        default=PROJECT_ROOT.parent / "Richter_2026_v2.h5",
                        help="Path to the Richter et al. (2026) HDF5 publication data.")
    parser.add_argument("--scan", action="store_true",
                        help="Rank all conditions by usable in-envelope points.")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--mechanism", choices=MECHANISMS, default="STA",
                        help="Chemical-kinetic mechanism whose 1-D solution is used.")
    parser.add_argument("--case", type=int, default=10,
                        help="Condition number (1-39). Default 10 = best ranked.")
    parser.add_argument("--make-profile", action="store_true")
    parser.add_argument("--length", type=float, default=1.0,
                        help="Domain length L [m] the flame slice is mapped onto.")
    parser.add_argument("--n-points", type=int, default=81)
    parser.add_argument("--suffix", type=int, default=1,
                        help="Sub-case index: writes Case5.<suffix>_L<L>m.npy")
    parser.add_argument("--rescale-x", action="store_true",
                        help="Keep M_r but renormalise X_H2O+X_NH3 to exactly 0.2.")
    parser.add_argument("--profiles-dir", type=Path,
                        default=PROJECT_ROOT / "input" / "profiles")
    parser.add_argument("--reports-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "real_flame")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = open_h5(args.h5)

    if args.scan or not (args.make_profile):
        rows = scan(root, args.mechanism, MR_MIN, MR_MAX)
        print_scan(rows, args.top)
        if not args.make_profile:
            return 0

    profile, info = build_profile(root, args.case, args.mechanism, args.length,
                                  args.n_points, args.rescale_x)

    print()
    print(f"--- Case 5.{args.suffix} profile ---")
    for key, value in info.items():
        print(f"  {key:>18}: {value}")

    npy_path = save_profile(profile, info, args.profiles_dir, args.reports_dir,
                            args.suffix)
    draw_profile(profile, info,
                 args.reports_dir / f"Case5.{args.suffix}_L{args.length:.1f}m.png")

    print()
    print("Next step (needs the local AbscDB and must be run from src/):")
    print(f"    cd src && python -c \"import gas_rte_solver as s; "
          f"s.getLBLsolution(300.0, 300.0, 1.0)\"   # once, to warm up")
    print(f"  then solve the reference for {npy_path.name} with the same routine "
          f"used for Cases 1.1-3.2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
