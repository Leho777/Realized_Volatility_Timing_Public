from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd

from realized_vol_timing.strategy import TimedCarryResult

if TYPE_CHECKING:
    from realized_vol_timing.experiments import CarryExperimentResult


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path


def prepare_output_dirs(
    base_dir: str | Path = "outputs",
    run_name: str = "spy_dynamic_carry",
    *,
    ensure_unique: bool = False,
) -> OutputPaths:
    root = Path(base_dir) / run_name
    if ensure_unique:
        base_root = root
        suffix = 2
        while root.exists():
            root = base_root.parent / f"{base_root.name}__{suffix:02d}"
            suffix += 1
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return OutputPaths(root=root, figures=figures, tables=tables)


def export_timed_carry_outputs(
    result: TimedCarryResult,
    *,
    base_dir: str | Path = "outputs",
    run_name: str = "spy_dynamic_carry",
    metadata: dict[str, Any] | None = None,
    ensure_unique: bool = False,
) -> OutputPaths:
    paths = prepare_output_dirs(
        base_dir=base_dir,
        run_name=run_name,
        ensure_unique=ensure_unique,
    )
    metadata = metadata or {}

    _export_tables(result, paths)
    _export_figures(result, paths)
    _export_metadata(result, paths, metadata)
    _export_summary(result, paths, metadata)
    return paths


def export_carry_experiment_outputs(
    result: "CarryExperimentResult",
    *,
    base_dir: str | Path = "outputs",
    run_name: str = "dynamic_carry",
    metadata: dict[str, Any] | None = None,
    proxy_metrics: pd.DataFrame | None = None,
    ensure_unique: bool = False,
) -> OutputPaths:
    paths = prepare_output_dirs(
        base_dir=base_dir,
        run_name=run_name,
        ensure_unique=ensure_unique,
    )
    metadata = metadata or {}

    _export_experiment_tables(result, paths, proxy_metrics)
    _export_experiment_figures(result, paths)
    _export_experiment_metadata(result, paths, metadata, proxy_metrics)
    _export_experiment_summary(result, paths, metadata, proxy_metrics)
    return paths


def _export_tables(result: TimedCarryResult, paths: OutputPaths) -> None:
    result.signal_result.signal_frame.to_csv(paths.tables / "signal_frame.csv", index=False)
    result.signal_result.parameter_frame.to_csv(
        paths.tables / "parameter_frame.csv",
        index=False,
    )
    if result.comparison is not None:
        result.comparison.reset_index().to_csv(paths.tables / "backtest_comparison.csv", index=False)
    if result.base_backtest is not None and result.timed_backtest is not None:
        nav_frame = (
            result.base_backtest.nav.rename(columns={"NAV": "base_nav"})
            .join(
                result.timed_backtest.nav.rename(columns={"NAV": "timed_nav"}),
                how="outer",
            )
            .reset_index()
            .rename(columns={"index": "date"})
        )
        nav_frame.to_csv(paths.tables / "nav_comparison.csv", index=False)


def _export_figures(result: TimedCarryResult, paths: OutputPaths) -> None:
    signal_frame = result.signal_result.signal_frame.dropna(subset=["spread"]).copy()
    parameter_frame = result.signal_result.parameter_frame.copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(signal_frame["date"], signal_frame["atm_implied_vol"], label="IV ATM", color="steelblue")
    axes[0].plot(
        signal_frame["date"],
        signal_frame["estimated_realized_vol"],
        label="Estimated realized vol",
        color="crimson",
        linestyle="--",
    )
    axes[0].set_title("Implied vs estimated realized volatility")
    axes[0].set_ylabel("Volatility")
    axes[0].legend()

    axes[1].fill_between(
        signal_frame["date"],
        0,
        signal_frame["spread"],
        where=(signal_frame["spread"] >= 0),
        color="green",
        alpha=0.25,
        label="Positive spread",
    )
    axes[1].fill_between(
        signal_frame["date"],
        0,
        signal_frame["spread"],
        where=(signal_frame["spread"] < 0),
        color="red",
        alpha=0.25,
        label="Negative spread",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Implied-realized spread")
    axes[1].set_ylabel("Spread")
    axes[1].legend()

    axes[2].plot(signal_frame["date"], signal_frame["allocation"], color="darkorange", linewidth=1.4)
    axes[2].axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_title("Dynamic allocation")
    axes[2].set_ylabel("Allocation")
    axes[2].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(paths.figures / "signal_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    param_names = ["kappa", "theta", "xi", "rho", "mu", "v0"]
    for ax, param_name in zip(axes.flat, param_names):
        ax.plot(parameter_frame["date"], parameter_frame[param_name], linewidth=1.1)
        ax.set_title(param_name)
    fig.suptitle("Rolling Heston parameters", y=1.02)
    fig.tight_layout()
    fig.savefig(paths.figures / "heston_parameters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if result.base_backtest is not None and result.timed_backtest is not None:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(result.base_backtest.nav.index, result.base_backtest.nav["NAV"], label="Base carry")
        ax.plot(result.timed_backtest.nav.index, result.timed_backtest.nav["NAV"], label="Timed carry")
        ax.set_title("NAV comparison")
        ax.set_ylabel("NAV")
        ax.set_xlabel("Date")
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.figures / "nav_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _export_metadata(
    result: TimedCarryResult,
    paths: OutputPaths,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "metadata": metadata,
        "files": {
            "figures": sorted(path.name for path in paths.figures.glob("*.png")),
            "tables": sorted(path.name for path in paths.tables.glob("*.csv")),
        },
        "row_counts": {
            "signal_frame": int(len(result.signal_result.signal_frame)),
            "parameter_frame": int(len(result.signal_result.parameter_frame)),
            "base_trades": int(len(result.base_trades)),
            "timed_trades": int(len(result.timed_trades)),
        },
    }
    (paths.root / "run_metadata.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _export_summary(
    result: TimedCarryResult,
    paths: OutputPaths,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "Timed carry backtest summary",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    if result.comparison is not None:
        lines.append(result.comparison.to_string())
    lines.append("")
    lines.append("Generated files:")
    for path in sorted(paths.figures.glob("*.png")):
        lines.append(f"  figure: {path.name}")
    for path in sorted(paths.tables.glob("*.csv")):
        lines.append(f"  table: {path.name}")
    (paths.root / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _export_experiment_tables(
    result: "CarryExperimentResult",
    paths: OutputPaths,
    proxy_metrics: pd.DataFrame | None,
) -> None:
    result.signal_frame.to_csv(paths.tables / "signal_frame_ukf.csv", index=False)
    result.parameter_frame.to_csv(paths.tables / "parameter_frame.csv", index=False)
    result.rv_signal_frame.to_csv(paths.tables / "signal_frame_rv_21d.csv", index=False)
    result.comparison_frame.reset_index().to_csv(
        paths.tables / "backtest_comparison.csv",
        index=False,
    )
    nav_frame = _build_experiment_nav_frame(result)
    nav_frame.to_csv(paths.tables / "nav_comparison.csv", index=False)
    if proxy_metrics is not None:
        proxy_metrics.reset_index().to_csv(paths.tables / "proxy_metrics.csv", index=False)


def _export_experiment_figures(
    result: "CarryExperimentResult",
    paths: OutputPaths,
) -> None:
    signal_frame = (
        result.signal_frame.merge(
            result.rv_signal_frame[
                ["date", "ticker", "benchmark_realized_vol", "spread", "allocation"]
            ].rename(
                columns={
                    "benchmark_realized_vol": "rv_21d_benchmark",
                    "spread": "spread_rv_21d",
                    "allocation": "allocation_rv_21d",
                }
            ),
            on=["date", "ticker"],
            how="left",
        )
        .dropna(subset=["spread"])
        .copy()
    )
    parameter_frame = result.parameter_frame.copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    axes[0].plot(signal_frame["date"], signal_frame["atm_implied_vol"], label="IV ATM", color="steelblue")
    axes[0].plot(
        signal_frame["date"],
        signal_frame["estimated_realized_vol"],
        label="UKF sigma_hat",
        color="crimson",
        linestyle="--",
    )
    axes[0].plot(signal_frame["date"], signal_frame["rv_21d"], label="RV 21d", color="darkgreen")
    axes[0].plot(signal_frame["date"], signal_frame["rv_63d"], label="RV 63d", color="purple", alpha=0.7)
    axes[0].set_title("Volatility proxies")
    axes[0].set_ylabel("Volatility")
    axes[0].legend()

    axes[1].plot(signal_frame["date"], signal_frame["spread"], color="darkorange", label="IV - UKF")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("UKF spread")
    axes[1].set_ylabel("Spread")
    axes[1].legend()

    axes[2].plot(
        signal_frame["date"],
        signal_frame["spread_rv_21d"],
        color="teal",
        label="IV - RV 21d",
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("RV 21d spread")
    axes[2].set_ylabel("Spread")
    axes[2].legend()

    axes[3].plot(signal_frame["date"], signal_frame["allocation"], label="Allocation UKF", color="darkorange")
    axes[3].plot(
        signal_frame["date"],
        signal_frame["allocation_rv_21d"],
        label="Allocation RV 21d",
        color="teal",
        alpha=0.85,
    )
    axes[3].axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    axes[3].set_title("Dynamic allocation comparison")
    axes[3].set_ylabel("Allocation")
    axes[3].set_xlabel("Date")
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(paths.figures / "signal_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    param_names = ["kappa", "theta", "xi", "rho", "mu", "v0"]
    for ax, param_name in zip(axes.flat, param_names):
        ax.plot(parameter_frame["date"], parameter_frame[param_name], linewidth=1.1)
        ax.set_title(param_name)
    fig.suptitle("Rolling Heston parameters", y=1.02)
    fig.tight_layout()
    fig.savefig(paths.figures / "heston_parameters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    nav_frame = _build_experiment_nav_frame(result).set_index("date")
    fig, ax = plt.subplots(figsize=(14, 5))
    for column, label in (
        ("base_nav", "Base carry"),
        ("ukf_timed_nav", "Timed carry UKF"),
        ("rv_21d_timed_nav", "Timed carry RV_21d"),
    ):
        ax.plot(nav_frame.index, nav_frame[column], label=label)
    ax.set_title("NAV comparison")
    ax.set_ylabel("NAV")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures / "nav_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _export_experiment_metadata(
    result: "CarryExperimentResult",
    paths: OutputPaths,
    metadata: dict[str, Any],
    proxy_metrics: pd.DataFrame | None,
) -> None:
    payload = {
        "metadata": metadata,
        "files": {
            "figures": sorted(path.name for path in paths.figures.glob("*.png")),
            "tables": sorted(path.name for path in paths.tables.glob("*.csv")),
        },
        "row_counts": {
            "signal_frame_ukf": int(len(result.signal_frame)),
            "signal_frame_rv_21d": int(len(result.rv_signal_frame)),
            "parameter_frame": int(len(result.parameter_frame)),
            "base_trades": int(len(result.base_trades)),
            "ukf_timed_trades": int(len(result.ukf_timed_trades)),
            "rv_timed_trades": int(len(result.rv_timed_trades)),
        },
        "proxy_metrics_rows": 0 if proxy_metrics is None else int(len(proxy_metrics)),
    }
    (paths.root / "run_metadata.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _export_experiment_summary(
    result: "CarryExperimentResult",
    paths: OutputPaths,
    metadata: dict[str, Any],
    proxy_metrics: pd.DataFrame | None,
) -> None:
    lines = [
        "Dynamic carry experiment summary",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Backtest comparison:")
    lines.append(result.comparison_frame.to_string())
    if proxy_metrics is not None:
        lines.append("")
        lines.append("Proxy metrics:")
        lines.append(proxy_metrics.to_string())
    lines.append("")
    lines.append("Generated files:")
    for path in sorted(paths.figures.glob("*.png")):
        lines.append(f"  figure: {path.name}")
    for path in sorted(paths.tables.glob("*.csv")):
        lines.append(f"  table: {path.name}")
    (paths.root / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _build_experiment_nav_frame(result: "CarryExperimentResult") -> pd.DataFrame:
    nav_frame = (
        result.base_backtest.nav.rename(columns={"NAV": "base_nav"})
        .join(
            result.ukf_timed_backtest.nav.rename(columns={"NAV": "ukf_timed_nav"}),
            how="outer",
        )
        .join(
            result.rv_timed_backtest.nav.rename(columns={"NAV": "rv_21d_timed_nav"}),
            how="outer",
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )
    return nav_frame
