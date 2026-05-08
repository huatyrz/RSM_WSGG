#!/usr/bin/env python3
"""Generate publication-ready validation figures without relying on notebooks."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, NullFormatter
from scipy.constants import sigma
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import gas_rte_solver as solver
from src.utils import read_from_excel


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wang = load_module_from_path("wang_wsgg", PROJECT_ROOT / "src" / "models" / "wang_wsgg.py")

DEFAULT_MAIN_DATASET = PROJECT_ROOT / "input" / "MixEmiss_p1.0_X0.2_Mr0.1-4.0_40_T40_path40.npy"
DEFAULT_VALIDATION_DATASET = PROJECT_ROOT / "input" / "MixEmiss_p1.0_X0.2_Mr0.1-4.0_15_T22_path40.npy"
DEFAULT_WANG_FILE = PROJECT_ROOT / "input" / "coefficients_from_wang.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "img" / "publication"
SERIES_COLORS = ["#4D4D4D", "#009E73", "#AA4499", "#E69F00", "#8C6D31", "#7A7A7A", "#66A61E", "#A6761D"]
COLORS = {
    "reference": "#111111",
    "present": "#C62828",
    "wang": "#1565C0",
    "zero": "#7F7F7F",
    "band": "#ECECEC",
    "guide": "#6E6E6E",
}
FIGURE_DPI = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate journal-style validation plots.")
    parser.add_argument("--groups", nargs="+", choices=["all", "parity", "emissivity", "cases"], default=["all"])
    parser.add_argument("--coefficients", type=Path, default=None)
    parser.add_argument("--main-dataset", type=Path, default=DEFAULT_MAIN_DATASET)
    parser.add_argument("--validation-dataset", type=Path, default=DEFAULT_VALIDATION_DATASET)
    parser.add_argument("--wang-file", type=Path, default=DEFAULT_WANG_FILE)
    parser.add_argument("--sheet-name", default="Px=0.2bar")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--Pt", type=float, default=1.0)
    parser.add_argument("--X", type=float, default=0.2)
    parser.add_argument("--T-ref", type=float, default=1000.0)
    parser.add_argument("--boundary-temperature", type=float, default=300.0)
    parser.add_argument("--panel-labels", action="store_true", help="Add panel labels such as (a), (b), ...")
    return parser.parse_args()


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
        }
    )


def latest_coefficients_file(results_dir: Path) -> Path:
    matches = sorted(results_dir.glob("coefficients_*.xlsx"), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No coefficients_*.xlsx file found in {results_dir}")
    return matches[-1]


def load_emissivity_cube(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(file_path, allow_pickle=True)
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, np.ndarray) and raw.shape == () and raw.dtype == object:
        data = raw.item()
    elif hasattr(raw, "item"):
        data = raw.item()
    else:
        raise TypeError(f"Unsupported emissivity data format in {file_path}")

    mole_keys = sorted(data.keys(), key=float)
    temperature_keys = sorted(next(iter(data.values())).keys(), key=float)
    path_values = sorted(
        {
            float(path_length)
            for mole_key in mole_keys
            for temperature_key in temperature_keys
            for path_length, _ in data[mole_key][temperature_key]
        }
    )
    path_index = {path_length: idx for idx, path_length in enumerate(path_values)}

    epsilon = np.zeros((len(mole_keys), len(temperature_keys), len(path_values)), dtype=float)
    for i, mole_key in enumerate(mole_keys):
        for j, temperature_key in enumerate(temperature_keys):
            for path_length, emissivity in data[mole_key][temperature_key]:
                epsilon[i, j, path_index[float(path_length)]] = float(emissivity)

    mole_ratios = np.asarray([float(key) for key in mole_keys], dtype=float)
    temperatures = np.asarray([float(key) for key in temperature_keys], dtype=float)
    path_lengths = np.asarray(path_values, dtype=float)
    return mole_ratios, temperatures, path_lengths, epsilon


def polynomial_features(temperature_ratio: float, mole_ratio: float) -> np.ndarray:
    return np.asarray(
        [
            (temperature_ratio**p) * (mole_ratio**q)
            for p in range(5)
            for q in range(5)
            if p + q <= 4
        ],
        dtype=float,
    )


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def get_k_a_ml(temperature_ratio: float, mole_ratio: float, beta: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = polynomial_features(temperature_ratio, mole_ratio)
    weights = softmax(beta @ features)
    powers = np.arange(gamma.shape[1] - 1, -1, -1, dtype=float)
    absorption = softplus(gamma @ (mole_ratio**powers))
    absorption[0] = 0.0
    return weights, absorption


def predict_rsm_emissivity_cube(
    mole_ratios: Sequence[float],
    temperatures: Sequence[float],
    path_lengths: Sequence[float],
    beta: np.ndarray,
    gamma: np.ndarray,
    pt: float,
    x_value: float,
    t_ref: float,
) -> np.ndarray:
    epsilon = np.zeros((len(mole_ratios), len(temperatures), len(path_lengths)), dtype=float)
    for i, mole_ratio in enumerate(mole_ratios):
        for j, temperature in enumerate(temperatures):
            a_coeffs, k_coeffs = get_k_a_ml(float(temperature) / t_ref, float(mole_ratio), beta, gamma)
            for k, path_length in enumerate(path_lengths):
                epsilon[i, j, k] = np.sum(a_coeffs * (1.0 - np.exp(-k_coeffs * pt * x_value * float(path_length))))
    return epsilon

def calculate_wang_emissivity_cube(
    wang_file: Path,
    sheet_name: str,
    mole_ratios: Sequence[float],
    temperatures: Sequence[float],
    path_lengths: Sequence[float],
    pt: float,
) -> np.ndarray:
    coefficients, range_dict = wang.read_coefficients_with_ranges(str(wang_file), sheet_name=sheet_name)
    epsilon = np.zeros((len(mole_ratios), len(temperatures), len(path_lengths)), dtype=float)
    for i, mole_ratio in enumerate(mole_ratios):
        for j, temperature in enumerate(temperatures):
            a_coeffs, k_coeffs = wang.get_k_a(float(temperature), float(mole_ratio), 4, coefficients, range_dict)
            for k, path_length in enumerate(path_lengths):
                epsilon[i, j, k] = np.sum(a_coeffs * (1.0 - np.exp(-k_coeffs * pt * float(path_length))))
    return epsilon


def nearest_index(values: Sequence[float], target: float) -> int:
    values = np.asarray(values, dtype=float)
    return int(np.nanargmin(np.abs(values - target)))


def relative_error_percent(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    denominator = max(float(np.max(np.abs(reference))), 1e-10)
    return (prediction - reference) / denominator * 100.0


def error_thresholds(reference: np.ndarray, prediction: np.ndarray, thresholds: Sequence[float] = (0.05, 0.10)) -> Dict[str, float]:
    denominator = np.clip(np.abs(reference), 1e-8, None)
    relative_error = np.abs(prediction - reference) / denominator
    return {f"within_{int(100 * threshold)}pct": float(np.mean(relative_error <= threshold) * 100.0) for threshold in thresholds}


def parity_statistics(reference: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    residual = prediction - reference
    ss_res = float(np.sum((reference - prediction) ** 2))
    ss_tot = float(np.sum((reference - np.mean(reference)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r_squared, "mbe": float(np.mean(residual)), "rmse": float(np.sqrt(np.mean(residual**2)))}


def choose_series_colors(count: int) -> List[str]:
    if count <= len(SERIES_COLORS):
        return SERIES_COLORS[:count]
    repeats = int(math.ceil(count / len(SERIES_COLORS)))
    return (SERIES_COLORS * repeats)[:count]


def style_axes(ax, grid_axis: str | None = None, logx: bool = False) -> None:
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if grid_axis is not None:
        ax.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def add_panel_label(ax, label: str, enabled: bool = False) -> None:
    if not enabled:
        return
    ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def parity_guide_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], color=COLORS["reference"], linewidth=1.5, label="1:1"),
        Line2D([0], [0], color=COLORS["guide"], linewidth=1.0, linestyle="--", label="±5%"),
        Line2D([0], [0], color=COLORS["guide"], linewidth=0.9, linestyle=":", label="±10%"),
    ]


def model_identity_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], color=COLORS["reference"], linewidth=1.5, label="LBL"),
        Line2D([0], [0], linestyle="None", marker="o", markerfacecolor="white", markeredgecolor=COLORS["present"], markeredgewidth=1.1, markersize=5.7, label="Present model"),
        Line2D([0], [0], linestyle="None", marker="s", markerfacecolor="white", markeredgecolor=COLORS["wang"], markeredgewidth=1.1, markersize=5.4, label="Wang et al. WSGGM"),
    ]


def draw_parity_guides(ax, limits: Tuple[float, float]) -> None:
    line = np.linspace(limits[0], limits[1], 200)
    ax.fill_between(line, 0.95 * line, 1.05 * line, color=COLORS["band"], zorder=0)
    ax.plot(line, line, color=COLORS["reference"], linewidth=1.5, zorder=1)
    ax.plot(line, 1.05 * line, color=COLORS["guide"], linewidth=1.0, linestyle="--", zorder=1)
    ax.plot(line, 0.95 * line, color=COLORS["guide"], linewidth=1.0, linestyle="--", zorder=1)
    ax.plot(line, 1.10 * line, color=COLORS["guide"], linewidth=0.9, linestyle=":", zorder=1)
    ax.plot(line, 0.90 * line, color=COLORS["guide"], linewidth=0.9, linestyle=":", zorder=1)


def annotate_box(ax, text: str, xy: Tuple[float, float], ha: str = "left", va: str = "top") -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BFBFBF", "alpha": 0.95},
    )


def emissivity_ylim(
    reference_curves: Sequence[np.ndarray],
    rsm_curves: Sequence[np.ndarray],
    wang_curves: Sequence[np.ndarray],
) -> Tuple[float, float]:
    finite_groups = [
        np.asarray(curve, dtype=float).ravel()
        for curve in [*reference_curves, *rsm_curves, *wang_curves]
    ]
    if not finite_groups:
        return 0.0, 1.0

    finite_values = np.concatenate(finite_groups)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    span = y_max - y_min
    padding = max(0.015, span * 0.08)
    y_min = max(0.0, y_min - padding)
    y_max = min(1.0, y_max + padding)

    if y_max - y_min < 0.08:
        center = 0.5 * (y_min + y_max)
        half_span = 0.04
        y_min = max(0.0, center - half_span)
        y_max = min(1.0, center + half_span)

    tick_step = 0.02 if (y_max - y_min) <= 0.2 else 0.05
    y_min = max(0.0, math.floor(y_min / tick_step) * tick_step)
    y_max = min(1.0, math.ceil(y_max / tick_step) * tick_step)
    if y_max <= y_min:
        y_max = min(1.0, y_min + tick_step)
    return y_min, y_max


def stats_text(label: str, reference: np.ndarray, prediction: np.ndarray) -> str:
    stats = parity_statistics(reference, prediction)
    thresholds = error_thresholds(reference, prediction)
    return "\n".join([
        label,
        rf"$R^2$ = {stats['r2']:.4f}",
        rf"RMSE = {stats['rmse']:.4f}",
        rf"MBE = {stats['mbe']:.4f}",
        rf"<=5%: {thresholds['within_5pct']:.1f}%",
        rf"<=10%: {thresholds['within_10pct']:.1f}%",
    ])


def value_slug(value: float, precision: int = 3) -> str:
    return f"{value:.{precision}f}".replace(".", "p")


def plot_rsm_parity(reference: np.ndarray, prediction: np.ndarray, output_dir: Path) -> None:
    x = prediction.ravel()
    y = reference.ravel()
    density = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    order = np.argsort(density)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    draw_parity_guides(ax, (0.0, 1.0))
    scatter = ax.scatter(x[order], y[order], c=density[order], cmap="cividis", s=10, linewidths=0.0, alpha=0.9, rasterized=True, zorder=2)
    style_axes(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted total emissivity")
    ax.set_ylabel("Reference total emissivity (LBL)")
    ax.legend(handles=parity_guide_handles(), loc="upper left")
    annotate_box(ax, stats_text("Present RSM-WSGGM", y, x), (0.98, 0.04), ha="right", va="bottom")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.02)
    colorbar.set_label("Point density")
    colorbar.ax.tick_params(direction="in")
    save_figure(fig, output_dir, "parity_rsm_kde")
    plt.close(fig)

def plot_model_parity_comparison(reference: np.ndarray, prediction_rsm: np.ndarray, prediction_wang: np.ndarray, output_dir: Path) -> None:
    y = reference.ravel()
    x_rsm = prediction_rsm.ravel()
    x_wang = prediction_wang.ravel()

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    draw_parity_guides(ax, (0.0, 1.0))
    ax.scatter(x_wang, y, s=10, marker="s", facecolors="none", edgecolors=COLORS["wang"], linewidths=0.45, alpha=0.35, rasterized=True, zorder=2)
    ax.scatter(x_rsm, y, s=10, marker="o", facecolors="none", edgecolors=COLORS["present"], linewidths=0.45, alpha=0.35, rasterized=True, zorder=3)
    style_axes(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted total emissivity")
    ax.set_ylabel("Reference total emissivity (LBL)")
    guide_legend = ax.legend(handles=parity_guide_handles(), loc="upper left")
    model_handles = [model_identity_handles()[1], model_identity_handles()[2]]
    model_legend = ax.legend(handles=model_handles, loc="lower right")
    ax.add_artist(guide_legend)
    ax.add_artist(model_legend)
    annotate_box(ax, stats_text("Present", y, x_rsm), (0.02, 0.44))
    annotate_box(ax, stats_text("Wang", y, x_wang), (0.98, 0.04), ha="right", va="bottom")
    save_figure(fig, output_dir, "parity_models_vs_lbl")
    plt.close(fig)

def plot_emissivity_family_figure(
    x_values: np.ndarray,
    reference_curves: Sequence[np.ndarray],
    rsm_curves: Sequence[np.ndarray],
    wang_curves: Sequence[np.ndarray],
    series_labels: Sequence[str],
    series_title: str,
    annotation_text: str,
    x_label: str,
    output_stem: str,
    output_dir: Path,
) -> None:
    colors = choose_series_colors(len(series_labels))
    y_min, y_max = emissivity_ylim(reference_curves, rsm_curves, wang_curves)
    fig, ax = plt.subplots(figsize=(5.5, 7.0))

    for color, lbl, rsm_curve, wang_curve in zip(colors, reference_curves, rsm_curves, wang_curves):
        ax.plot(x_values, lbl, color=color, linewidth=1.45, zorder=1)
        ax.scatter(
            x_values,
            rsm_curve,
            marker="o",
            s=24,
            facecolors="white",
            edgecolors=COLORS["present"],
            linewidths=1.0,
            zorder=3,
        )
        ax.scatter(
            x_values,
            wang_curve,
            marker="s",
            s=21,
            facecolors="white",
            edgecolors=COLORS["wang"],
            linewidths=0.95,
            zorder=3,
        )

    style_axes(ax, grid_axis="y", logx=False)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(float(x_values.min()), float(x_values.max()))
    ax.set_ylabel(r"Total emissivity, $\varepsilon$")
    ax.set_xlabel(x_label)
    annotate_box(ax, annotation_text, (0.97, 0.96), ha="right")

    series_handles = [
        Line2D([0], [0], color=color, linewidth=1.6, label=label)
        for color, label in zip(colors, series_labels)
    ]
    legend_series = ax.legend(
        handles=series_handles,
        title=series_title,
        loc="upper left",
        ncol=2,
        columnspacing=0.9,
        handlelength=2.2,
    )
    legend_models = ax.legend(handles=model_identity_handles(), loc="lower right")
    ax.add_artist(legend_series)
    ax.add_artist(legend_models)

    save_figure(fig, output_dir, output_stem)
    plt.close(fig)


def plot_emissivity_vs_temperature(
    mole_ratios: np.ndarray,
    temperatures: np.ndarray,
    path_lengths: np.ndarray,
    reference: np.ndarray,
    prediction_rsm: np.ndarray,
    prediction_wang: np.ndarray,
    output_dir: Path,
) -> None:
    temperature_order = np.argsort(temperatures)
    temperatures_sorted = temperatures[temperature_order]
    path_order = np.argsort(path_lengths)
    path_lengths_sorted = path_lengths[path_order]
    selected_paths = [0.2, 1.0, 5.0, 10.0, 30.0, 60.0]
    selected_mole_ratios = [0.5, 3.0]
    selected_path_indices = [nearest_index(path_lengths_sorted, path) for path in selected_paths]
    selected_mole_indices = [nearest_index(mole_ratios, mr) for mr in selected_mole_ratios]

    for mr_value, mr_index in zip(selected_mole_ratios, selected_mole_indices):
        reference_curves = [reference[mr_index, temperature_order, idx] for idx in selected_path_indices]
        rsm_curves = [prediction_rsm[mr_index, temperature_order, idx] for idx in selected_path_indices]
        wang_curves = [prediction_wang[mr_index, temperature_order, idx] for idx in selected_path_indices]
        series_labels = [f"{path_lengths_sorted[idx]:.1f}" for idx in selected_path_indices]
        plot_emissivity_family_figure(
            temperatures_sorted,
            reference_curves,
            rsm_curves,
            wang_curves,
            series_labels,
            r"$P_tL$ (bar m)",
            rf"$M_r = {mr_value:.2f}$",
            r"Temperature, $T$ (K)",
            f"emissivity_vs_temperature_mr_{value_slug(mr_value)}",
            output_dir,
        )


def plot_emissivity_vs_path_length_by_temperature(
    mole_ratios: np.ndarray,
    temperatures: np.ndarray,
    path_lengths: np.ndarray,
    reference: np.ndarray,
    prediction_rsm: np.ndarray,
    prediction_wang: np.ndarray,
    output_dir: Path,
) -> None:
    path_order = np.argsort(path_lengths)
    path_lengths_sorted = path_lengths[path_order]
    selected_temperatures = [300.0, 700.0, 1100.0, 1500.0, 2100.0, 2400.0]
    selected_mole_ratios = [0.5, 3.0]
    selected_temperature_indices = [nearest_index(temperatures, temperature) for temperature in selected_temperatures]
    selected_mole_indices = [nearest_index(mole_ratios, mr) for mr in selected_mole_ratios]

    for mr_value, mr_index in zip(selected_mole_ratios, selected_mole_indices):
        reference_curves = [reference[mr_index, idx, :][path_order] for idx in selected_temperature_indices]
        rsm_curves = [prediction_rsm[mr_index, idx, :][path_order] for idx in selected_temperature_indices]
        wang_curves = [prediction_wang[mr_index, idx, :][path_order] for idx in selected_temperature_indices]
        series_labels = [f"{temperatures[idx]:.0f}" for idx in selected_temperature_indices]
        plot_emissivity_family_figure(
            path_lengths_sorted,
            reference_curves,
            rsm_curves,
            wang_curves,
            series_labels,
            r"$T$ (K)",
            rf"$M_r = {mr_value:.2f}$",
            r"Path length, $L$ (m)",
            f"emissivity_vs_path_length_mr_{value_slug(mr_value)}",
            output_dir,
        )


def plot_emissivity_vs_path_length_by_mr(
    mole_ratios: np.ndarray,
    temperatures: np.ndarray,
    path_lengths: np.ndarray,
    reference: np.ndarray,
    prediction_rsm: np.ndarray,
    prediction_wang: np.ndarray,
    output_dir: Path,
) -> None:
    path_order = np.argsort(path_lengths)
    path_lengths_sorted = path_lengths[path_order]
    selected_temperatures = [1100.0, 1700.0]
    selected_mole_ratios = [0.1, 0.5, 1.0, 2.0, 3.5, 4.0]
    selected_temperature_indices = [nearest_index(temperatures, temperature) for temperature in selected_temperatures]
    selected_mole_indices = [nearest_index(mole_ratios, mr) for mr in selected_mole_ratios]

    for temperature_index in selected_temperature_indices:
        temperature_value = temperatures[temperature_index]
        reference_curves = [reference[idx, temperature_index, :][path_order] for idx in selected_mole_indices]
        rsm_curves = [prediction_rsm[idx, temperature_index, :][path_order] for idx in selected_mole_indices]
        wang_curves = [prediction_wang[idx, temperature_index, :][path_order] for idx in selected_mole_indices]
        series_labels = [f"{mr_value:.1f}" for mr_value in selected_mole_ratios]
        plot_emissivity_family_figure(
            path_lengths_sorted,
            reference_curves,
            rsm_curves,
            wang_curves,
            series_labels,
            r"$M_r$ (-)",
            rf"$T = {temperature_value:.0f}$ K",
            r"Path length, $L$ (m)",
            f"emissivity_vs_path_length_t_{int(round(temperature_value))}K",
            output_dir,
        )


def wsgg_rsm_case_solution(
    xs: np.ndarray,
    temperature_profile: np.ndarray,
    x_h2o: np.ndarray,
    x_nh3: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    t_ref: float,
    boundary_temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    gray_count = beta.shape[0]
    weights = np.zeros((len(xs), gray_count), dtype=float)
    absorption = np.zeros((len(xs), gray_count), dtype=float)
    for index in range(len(xs)):
        weights[index], absorption[index] = get_k_a_ml(temperature_profile[index] / t_ref, float(x_h2o[index] / x_nh3[index]), beta, gamma)
    absorption *= (x_h2o + x_nh3).reshape(-1, 1)
    total_q = np.zeros(len(xs), dtype=float)
    total_dq = np.zeros(len(xs), dtype=float)
    for gas_index in range(gray_count):
        i_b = sigma * np.power(temperature_profile, 4) * weights[:, gas_index] / np.pi
        i_b_left = sigma * boundary_temperature**4 * weights[0, gas_index] / np.pi
        i_b_right = sigma * boundary_temperature**4 * weights[-1, gas_index] / np.pi
        _, q, dq = solver.solve_rte(xs, absorption[:, gas_index], i_b, i_b_left, i_b_right)
        total_q += q
        total_dq += dq
    return total_q, total_dq


def wsgg_wang_case_solution(
    xs: np.ndarray,
    temperature_profile: np.ndarray,
    x_h2o: np.ndarray,
    x_nh3: np.ndarray,
    wang_file: Path,
    sheet_name: str,
    boundary_temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    coefficients, range_dict = wang.read_coefficients_with_ranges(str(wang_file), sheet_name=sheet_name)
    weights = np.zeros((len(xs), 5), dtype=float)
    absorption = np.zeros((len(xs), 5), dtype=float)
    for index in range(len(xs)):
        weights[index], absorption[index] = wang.get_k_a(float(temperature_profile[index]), float(x_h2o[index] / x_nh3[index]), 4, coefficients, range_dict)
    total_q = np.zeros(len(xs), dtype=float)
    total_dq = np.zeros(len(xs), dtype=float)
    for gas_index in range(5):
        i_b = sigma * np.power(temperature_profile, 4) * weights[:, gas_index] / np.pi
        i_b_left = sigma * boundary_temperature**4 * weights[0, gas_index] / np.pi
        i_b_right = sigma * boundary_temperature**4 * weights[-1, gas_index] / np.pi
        _, q, dq = solver.solve_rte(xs, absorption[:, gas_index], i_b, i_b_left, i_b_right)
        total_q += q
        total_dq += dq
    return total_q, total_dq


def case_sort_key(path: Path) -> Tuple[int, int, float]:
    case_name, length_tag = path.stem.split("_")
    case_major, case_minor = case_name.replace("Case", "").split(".")
    return int(case_major), int(case_minor), float(length_tag[1:-1])


def rounded_error_limit(*arrays: np.ndarray) -> float:
    max_error = max(float(np.max(np.abs(array))) for array in arrays)
    return max(5.0, math.ceil(max_error / 5.0) * 5.0)


def plot_case_validation(
    beta: np.ndarray,
    gamma: np.ndarray,
    wang_file: Path,
    sheet_name: str,
    t_ref: float,
    boundary_temperature: float,
    output_dir: Path,
    panel_labels: bool = False,
) -> List[Dict[str, float]]:
    summary_rows: List[Dict[str, float]] = []
    case_files = sorted((PROJECT_ROOT / "input" / "profiles").glob("Case*.npy"), key=case_sort_key)

    for config_index, file_path in enumerate(case_files, start=1):
        filename = file_path.stem
        _, length_tag = filename.split("_")
        length_value = float(length_tag[1:-1])
        profile = np.load(file_path, allow_pickle=False)
        reference_results = np.load(PROJECT_ROOT / "results" / f"{filename}.npy") / 1e3
        xs = profile["x"]
        x_rel = xs / length_value if length_value > 0 else xs
        temperature_profile = profile["T"]
        x_h2o = profile["Y_H2O"]
        x_nh3 = profile["Y_NH3"] + 1e-4

        wang_q, wang_dq = wsgg_wang_case_solution(xs, temperature_profile, x_h2o, x_nh3, wang_file, sheet_name, boundary_temperature)
        rsm_q, rsm_dq = wsgg_rsm_case_solution(xs, temperature_profile, x_h2o, x_nh3, beta, gamma, t_ref, boundary_temperature)
        wang_q, wang_dq = wang_q / 1e3, wang_dq / 1e3
        rsm_q, rsm_dq = rsm_q / 1e3, rsm_dq / 1e3

        reference_q = reference_results[:, 1]
        reference_st = -reference_results[:, 2]
        delta_q_rsm = relative_error_percent(reference_q, rsm_q)
        delta_q_wang = relative_error_percent(reference_q, wang_q)
        delta_st_rsm = relative_error_percent(reference_st, -rsm_dq)
        delta_st_wang = relative_error_percent(reference_st, -wang_dq)
        error_limit = rounded_error_limit(delta_q_rsm, delta_q_wang, delta_st_rsm, delta_st_wang)

        fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex="col", gridspec_kw={"height_ratios": [3.1, 1.2], "wspace": 0.24, "hspace": 0.08})
        (ax_q, ax_st), (ax_q_err, ax_st_err) = axes

        ax_q.plot(x_rel, reference_q, color=COLORS["reference"], linewidth=1.8, label="LBL")
        ax_q.plot(x_rel, rsm_q, color=COLORS["present"], linewidth=1.25, linestyle="--")
        ax_q.plot(x_rel, wang_q, color=COLORS["wang"], linewidth=1.2, linestyle=":")
        ax_q.scatter(x_rel[::2], rsm_q[::2], marker="o", s=18, facecolors="white", edgecolors=COLORS["present"], linewidths=0.95, zorder=3)
        ax_q.scatter(x_rel[::2], wang_q[::2], marker="s", s=16, facecolors="white", edgecolors=COLORS["wang"], linewidths=0.9, zorder=3)
        ax_q.set_ylabel("Radiative heat flux\n(kW m$^{-2}$)")
        ax_q.set_xlim(0.0, 1.0)
        style_axes(ax_q, grid_axis="y")
        add_panel_label(ax_q, "(a)", panel_labels)
        annotate_box(ax_q, rf"Configuration {config_index}\n$L = {length_value:.1f}$ m", (0.97, 0.96), ha="right")
        plt.setp(ax_q.get_xticklabels(), visible=False)

        ax_q_err.axhline(0.0, color=COLORS["zero"], linewidth=0.9, linestyle="--")
        ax_q_err.plot(x_rel, delta_q_rsm, color=COLORS["present"], linewidth=1.15, linestyle="--")
        ax_q_err.plot(x_rel, delta_q_wang, color=COLORS["wang"], linewidth=1.1, linestyle=":")
        ax_q_err.set_ylabel("Error (%)")
        ax_q_err.set_xlabel(r"Normalised position, $x/L$")
        ax_q_err.set_xlim(0.0, 1.0)
        ax_q_err.set_ylim(-error_limit, error_limit)
        style_axes(ax_q_err, grid_axis="y")
        add_panel_label(ax_q_err, "(b)", panel_labels)

        ax_st.plot(x_rel, reference_st, color=COLORS["reference"], linewidth=1.8)
        ax_st.plot(x_rel, -rsm_dq, color=COLORS["present"], linewidth=1.25, linestyle="--")
        ax_st.plot(x_rel, -wang_dq, color=COLORS["wang"], linewidth=1.2, linestyle=":")
        ax_st.scatter(x_rel[::2], (-rsm_dq)[::2], marker="o", s=18, facecolors="white", edgecolors=COLORS["present"], linewidths=0.95, zorder=3)
        ax_st.scatter(x_rel[::2], (-wang_dq)[::2], marker="s", s=16, facecolors="white", edgecolors=COLORS["wang"], linewidths=0.9, zorder=3)
        ax_st.set_ylabel("Source term\n(kW m$^{-3}$)")
        ax_st.set_xlim(0.0, 1.0)
        style_axes(ax_st, grid_axis="y")
        add_panel_label(ax_st, "(c)", panel_labels)
        plt.setp(ax_st.get_xticklabels(), visible=False)

        ax_st_err.axhline(0.0, color=COLORS["zero"], linewidth=0.9, linestyle="--")
        ax_st_err.plot(x_rel, delta_st_rsm, color=COLORS["present"], linewidth=1.15, linestyle="--")
        ax_st_err.plot(x_rel, delta_st_wang, color=COLORS["wang"], linewidth=1.1, linestyle=":")
        ax_st_err.set_ylabel("Error (%)")
        ax_st_err.set_xlabel(r"Normalised position, $x/L$")
        ax_st_err.set_xlim(0.0, 1.0)
        ax_st_err.set_ylim(-error_limit, error_limit)
        style_axes(ax_st_err, grid_axis="y")
        add_panel_label(ax_st_err, "(d)", panel_labels)

        fig.legend(handles=[
            Line2D([0], [0], color=COLORS["reference"], linewidth=1.6, label="LBL"),
            Line2D([0], [0], color=COLORS["present"], linewidth=1.0, linestyle="--", marker="o", markerfacecolor="white", markeredgecolor=COLORS["present"], markersize=4.8, label="Present model"),
            Line2D([0], [0], color=COLORS["wang"], linewidth=1.0, linestyle=":", marker="s", markerfacecolor="white", markeredgecolor=COLORS["wang"], markersize=4.6, label="Wang et al. WSGGM"),
        ], loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3)
        save_figure(fig, output_dir, f"case_validation_{filename}")
        plt.close(fig)

        summary_rows.append({
            "config_label": f"Cfg{config_index}",
            "source_name": filename,
            "wang_mae_q": float(np.mean(np.abs(delta_q_wang))),
            "wang_maxae_q": float(np.max(np.abs(delta_q_wang))),
            "wang_mae_st": float(np.mean(np.abs(delta_st_wang))),
            "wang_maxae_st": float(np.max(np.abs(delta_st_wang))),
            "rsm_mae_q": float(np.mean(np.abs(delta_q_rsm))),
            "rsm_maxae_q": float(np.max(np.abs(delta_q_rsm))),
            "rsm_mae_st": float(np.mean(np.abs(delta_st_rsm))),
            "rsm_maxae_st": float(np.max(np.abs(delta_st_rsm))),
        })
        print(f"Cfg{config_index} -> {filename}")
    return summary_rows


def plot_case_error_summary(summary_rows: List[Dict[str, float]], output_dir: Path, panel_labels: bool = False) -> None:
    if not summary_rows:
        return

    case_labels = [row["config_label"] for row in summary_rows]
    x = np.arange(len(summary_rows))
    width = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.28})

    plot_specs = [
        (axes[0], "rsm_mae_q", "rsm_maxae_q", "wang_mae_q", "wang_maxae_q", "Radiative heat flux", "(a)"),
        (axes[1], "rsm_mae_st", "rsm_maxae_st", "wang_mae_st", "wang_maxae_st", "Source term", "(b)"),
    ]

    for ax, rsm_mae_key, rsm_max_key, wang_mae_key, wang_max_key, title, panel_label in plot_specs:
        wang_mae = np.asarray([row[wang_mae_key] for row in summary_rows], dtype=float)
        wang_cap = np.asarray([row[wang_max_key] for row in summary_rows], dtype=float)
        rsm_mae = np.asarray([row[rsm_mae_key] for row in summary_rows], dtype=float)
        rsm_cap = np.asarray([row[rsm_max_key] for row in summary_rows], dtype=float)

        ax.bar(x - width / 2, wang_mae, width, color=COLORS["wang"], alpha=0.88, edgecolor=COLORS["wang"], linewidth=0.9)
        ax.bar(x - width / 2, np.maximum(0.0, wang_cap - wang_mae), width, bottom=wang_mae, facecolor="white", edgecolor=COLORS["wang"], hatch="///", linewidth=0.9)
        ax.bar(x + width / 2, rsm_mae, width, color=COLORS["present"], alpha=0.88, edgecolor=COLORS["present"], linewidth=0.9)
        ax.bar(x + width / 2, np.maximum(0.0, rsm_cap - rsm_mae), width, bottom=rsm_mae, facecolor="white", edgecolor=COLORS["present"], hatch="///", linewidth=0.9)

        for x_coord, total in zip(x - width / 2, wang_cap):
            ax.text(x_coord, total + 0.6, f"{total:.1f}", ha="center", va="bottom", fontsize=8.2, color=COLORS["wang"])
        for x_coord, total in zip(x + width / 2, rsm_cap):
            ax.text(x_coord, total + 0.6, f"{total:.1f}", ha="center", va="bottom", fontsize=8.2, color=COLORS["present"])

        style_axes(ax, grid_axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels)
        ax.set_xlabel("Configuration")
        ax.set_title(title)
        ax.set_ylabel("Normalised error (%)")
        ax.set_ylim(0.0, max(float(np.max(wang_cap)), float(np.max(rsm_cap))) * 1.18)
        add_panel_label(ax, panel_label, panel_labels)

    legend_handles = [
        Patch(facecolor=COLORS["wang"], edgecolor=COLORS["wang"], label="Wang MAE"),
        Patch(facecolor="white", edgecolor=COLORS["wang"], hatch="///", label="Wang MaxAE"),
        Patch(facecolor=COLORS["present"], edgecolor=COLORS["present"], label="Present MAE"),
        Patch(facecolor="white", edgecolor=COLORS["present"], hatch="///", label="Present MaxAE"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=4)
    save_figure(fig, output_dir, "case_error_summary")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_publication_style()
    coeff_file = args.coefficients or latest_coefficients_file(PROJECT_ROOT / "results")
    beta, gamma = read_from_excel(str(coeff_file))
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    output_dir = args.output_dir
    groups = set(args.groups)
    if "all" in groups:
        groups = {"parity", "emissivity", "cases"}

    print(f"Using coefficient file: {coeff_file}")
    print(f"Output directory: {output_dir}")

    if "parity" in groups or "emissivity" in groups:
        main_mr, main_t, main_l, main_ref = load_emissivity_cube(args.main_dataset)
        main_rsm = predict_rsm_emissivity_cube(main_mr, main_t, main_l, beta, gamma, args.Pt, args.X, args.T_ref)
        main_wang = calculate_wang_emissivity_cube(args.wang_file, args.sheet_name, main_mr, main_t, main_l, args.Pt)

    if "parity" in groups:
        plot_rsm_parity(main_ref, main_rsm, output_dir)
        plot_model_parity_comparison(main_ref, main_rsm, main_wang, output_dir)

    if "emissivity" in groups:
        plot_emissivity_vs_temperature(main_mr, main_t, main_l, main_ref, main_rsm, main_wang, output_dir)
        val_mr, val_t, val_l, val_ref = load_emissivity_cube(args.validation_dataset)
        val_rsm = predict_rsm_emissivity_cube(val_mr, val_t, val_l, beta, gamma, args.Pt, args.X, args.T_ref)
        val_wang = calculate_wang_emissivity_cube(args.wang_file, args.sheet_name, val_mr, val_t, val_l, args.Pt)
        plot_emissivity_vs_path_length_by_temperature(val_mr, val_t, val_l, val_ref, val_rsm, val_wang, output_dir)
        plot_emissivity_vs_path_length_by_mr(val_mr, val_t, val_l, val_ref, val_rsm, val_wang, output_dir)

    if "cases" in groups:
        summary_rows = plot_case_validation(beta, gamma, args.wang_file, args.sheet_name, args.T_ref, args.boundary_temperature, output_dir, panel_labels=args.panel_labels)
        plot_case_error_summary(summary_rows, output_dir, panel_labels=args.panel_labels)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

