from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realized_vol_timing import (
    compute_proxy_metrics,
    export_carry_experiment_outputs,
    resolve_experiment_dates,
    run_carry_experiment,
)
from realized_vol_timing.config import AllocationConfig, HestonUKFConfig, RollingWindowConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dynamic carry experiment on real data for SPY or AAPL, "
            "compare UKF timing vs RV_21d timing, and export figures/tables."
        )
    )
    parser.add_argument("--ticker", choices=["SPY", "AAPL"], default="SPY")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--full-sample", action="store_true")
    parser.add_argument("--optimizer-maxiter", type=int, default=12)
    parser.add_argument("--window", type=int, default=42)
    parser.add_argument("--refit-every", type=int, default=10)
    parser.add_argument("--base-allocation", type=float, default=1.0)
    parser.add_argument("--sensitivity", type=float, default=0.4)
    parser.add_argument("--implied-vol-maturity-days", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/dynamic_carry")
    parser.add_argument("--run-name")
    parser.add_argument("--no-save-outputs", action="store_true")
    return parser.parse_args()


def _resolve_dates(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.start_date or args.end_date:
        if not (args.start_date and args.end_date):
            raise SystemExit("Provide both --start-date and --end-date, or neither.")
        return datetime.fromisoformat(args.start_date), datetime.fromisoformat(args.end_date)
    return resolve_experiment_dates(args.ticker, full_sample=args.full_sample)


def _default_run_name(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    *,
    full_sample: bool,
) -> str:
    mode_label = "full_sample" if full_sample else "date_range"
    return (
        f"{ticker.lower()}_dynamic_carry_"
        f"{mode_label}_{start_date.date().isoformat()}_{end_date.date().isoformat()}"
    )


def main() -> None:
    args = _parse_args()
    start_date, end_date = _resolve_dates(args)

    try:
        experiment = run_carry_experiment(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            implied_vol_maturity_days=args.implied_vol_maturity_days,
            ukf_config=HestonUKFConfig(optimizer_maxiter=args.optimizer_maxiter),
            rolling_config=RollingWindowConfig(
                window=args.window,
                refit_every=args.refit_every,
            ),
            allocation_config=AllocationConfig(
                base_allocation=args.base_allocation,
                sensitivity=args.sensitivity,
            ),
        )
    except ValueError as exc:
        print(f"Impossible de lancer l'experience: {exc}")
        print(
            "Essaie une plage de dates plus longue ou reduis --window / --refit-every "
            "si tu es en mode debug."
        )
        return
    proxy_metrics = compute_proxy_metrics(experiment.signal_frame)

    mode_label = "full sample" if args.full_sample and not (args.start_date or args.end_date) else "custom range"
    print(
        f"ticker={args.ticker} | mode={mode_label} | start={start_date.date()} | "
        f"end={end_date.date()} | window={args.window} | refit_every={args.refit_every} | "
        f"optimizer_maxiter={args.optimizer_maxiter}"
    )
    print("")
    print("Proxy metrics:")
    print(proxy_metrics.to_string())
    print("")
    print("Backtest comparison:")
    print(experiment.comparison_frame.to_string())

    if args.no_save_outputs:
        return

    run_name = args.run_name or _default_run_name(
        args.ticker,
        start_date,
        end_date,
        full_sample=args.full_sample and not (args.start_date or args.end_date),
    )
    output_paths = export_carry_experiment_outputs(
        experiment,
        base_dir=args.output_dir,
        run_name=run_name,
        proxy_metrics=proxy_metrics,
        ensure_unique=True,
        metadata={
            "ticker": args.ticker,
            "mode": mode_label,
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "window": args.window,
            "refit_every": args.refit_every,
            "optimizer_maxiter": args.optimizer_maxiter,
            "base_allocation": args.base_allocation,
            "sensitivity": args.sensitivity,
            "implied_vol_maturity_days": args.implied_vol_maturity_days,
        },
    )
    print("")
    print(f"Outputs saved to: {output_paths.root.resolve()}")


if __name__ == "__main__":
    main()
