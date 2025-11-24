#!/usr/bin/env python3

"""Benchmark training speed across GPU counts and PXRD conditioning modes.

Generates per-run configs, launches ``bin/train.py`` (with ``torchrun`` when
needed), scrapes iteration timings from stdout, and writes JSON/LaTeX
summaries for the paper throughput table.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import re
import signal
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import shlex
import yaml


ITERATION_REGEX = re.compile(r"iter\s+(\d+):\s+loss\s+([0-9.]+),\s+time\s+([0-9.]+)ms")


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as a boolean.")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dump_yaml(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


@dataclass
class BenchmarkResult:
    run_name: str
    gpu_count: int
    enhanced_pxrd_conditioning: bool
    batch_size: int
    grad_accumulation: int
    iteration_mean_s: float
    iteration_std_s: float
    throughput_mean: float
    throughput_std: float
    speedup: float = math.nan

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accumulation * self.gpu_count

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "gpu_count": self.gpu_count,
            "enhanced_pxrd_conditioning": self.enhanced_pxrd_conditioning,
            "batch_size": self.batch_size,
            "grad_accumulation": self.grad_accumulation,
            "iteration_mean_s": self.iteration_mean_s,
            "iteration_std_s": self.iteration_std_s,
            "throughput_mean": self.throughput_mean,
            "throughput_std": self.throughput_std,
            "speedup": self.speedup,
        }


def _format_time(mean_s: float, std_s: float) -> str:
    return f"${mean_s:.2f} \\pm {std_s:.2f}$"


def _format_throughput(mean_val: float, std_val: float) -> str:
    return f"${mean_val:.0f} \\pm {std_val:.0f}$"


def _describe_run(gpu_count: int, enhanced: bool) -> str:
    gpu_label = f"{gpu_count} GPU" + ("s" if gpu_count > 1 else "")
    pxrd_label = "enhanced" if enhanced else "baseline"
    return f"{gpu_label} ({pxrd_label})"


def _pxrd_label(enhanced: bool) -> str:
    return "Device" if enhanced else "Host"


def _build_command(*, gpu_count: int, config_path: Path, force_torchrun: bool) -> List[str]:
    if gpu_count == 1 and not force_torchrun:
        return [sys.executable, "bin/train.py", "--config", str(config_path)]
    return [
        "torchrun",
        "--nproc_per_node",
        str(gpu_count),
        "bin/train.py",
        "--config",
        str(config_path),
    ]


def _stream_training(command: Sequence[str], cutoff_iters: int | None = None) -> List[float]:
    iteration_times: List[float] = []
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    requested_stop = False
    assert process.stdout is not None
    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            print(line)
            match = ITERATION_REGEX.search(line)
            if match:
                time_ms = float(match.group(3))
                iteration_times.append(time_ms / 1000.0)
                if cutoff_iters is not None:
                    iter_index = int(match.group(1))
                    if not requested_stop and (iter_index + 1) >= cutoff_iters:
                        requested_stop = True
                        print(f"[info] Cutoff reached at iteration {iter_index}, sending SIGINT.")
                        process.send_signal(signal.SIGINT)
    finally:
        process.wait()
    if process.returncode != 0:
        expected_sigint_codes = {-signal.SIGINT.value, 128 + signal.SIGINT.value}
        if not (requested_stop and process.returncode in expected_sigint_codes):
            raise subprocess.CalledProcessError(process.returncode, command)
    return iteration_times


def _select_samples(samples: Sequence[float], warmup: int, measure: int) -> List[float]:
    if warmup >= len(samples):
        return []
    window = samples[warmup : warmup + measure if measure > 0 else None]
    return list(window)


def _summarise_iteration_times(times_s: Sequence[float]) -> tuple[float, float]:
    if not times_s:
        return math.nan, math.nan
    if len(times_s) == 1:
        return times_s[0], 0.0
    return statistics.mean(times_s), statistics.pstdev(times_s)


def _summarise_throughput(times_s: Sequence[float], batch: int) -> tuple[float, float]:
    if not times_s:
        return math.nan, math.nan
    per_iter = [batch / t for t in times_s if t > 0]
    if not per_iter:
        return math.nan, math.nan
    if len(per_iter) == 1:
        return per_iter[0], 0.0
    return statistics.mean(per_iter), statistics.pstdev(per_iter)


def _write_json(results: Sequence[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([result.to_dict() for result in results], handle, indent=2)


def _write_latex(results: Sequence[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            batch_str = f"{result.batch_size}\\times{result.grad_accumulation}\\times{result.gpu_count}"
            line = (
                f"{result.run_name} & {result.gpu_count} & {_pxrd_label(result.enhanced_pxrd_conditioning)} & "
                f"{_format_time(result.iteration_mean_s, result.iteration_std_s)} & {batch_str} & "
                f"{_format_throughput(result.throughput_mean, result.throughput_std)} & "
                f"${result.speedup:.1f}\\times$ "
                "\\\\\n"
            )
            handle.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"), help="Base YAML config")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/train_benchmarks"),
        help="Folder that collects generated configs/checkpoints.",
    )
    parser.add_argument(
        "--gpu-counts",
        type=int,
        nargs="+",
        default=[1, 2],
        help="GPU counts to benchmark (default: 1 2).",
    )
    parser.add_argument(
        "--pxrd-modes",
        type=str,
        nargs="+",
        default=["false", "true"],
        help="Values for enhanced_pxrd_conditioning (accepts booleans).",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of initial iterations excluded from statistics.",
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=50000,
        help="Number of iterations used for statistics after the warm-up window.",
    )
    parser.add_argument(
        "--cutoff-iters",
        type=int,
        default=100,
        help="Total iterations to run before simulating an unexpected stop.",
    )
    parser.add_argument(
        "--force-torchrun",
        action="store_true",
        help="Always launch via torchrun, even for single-GPU benchmarks.",
    )
    parser.add_argument(
        "--train-extra-args",
        type=str,
        default="",
        help="Additional CLI arguments appended after the train.py invocation.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("runs/train_benchmarks/summary.json"),
        help="Destination for the JSON summary report.",
    )
    parser.add_argument(
        "--latex-path",
        type=Path,
        default=Path("docs/tables/train_optimisation_rows.tex"),
        help="Path of the LaTeX snippet that receives the formatted rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without launching training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = _load_config(args.config)
    pxrd_options = [_str_to_bool(value) for value in args.pxrd_modes]
    combos = list(itertools.product(sorted(set(args.gpu_counts)), pxrd_options))

    run_results: List[BenchmarkResult] = []

    for gpu_count, enhanced in combos:
        run_name = f"g{gpu_count}_pxrd_{'on' if enhanced else 'off'}"
        human_label = _describe_run(gpu_count, enhanced)
        run_dir = args.out_root / run_name
        config_path = run_dir / "config.yaml"

        config = copy.deepcopy(base_config)
        config["out_dir"] = str(run_dir)
        config["tensorboard_log_dir"] = str(run_dir / "tensorboard")
        config["enhanced_pxrd_conditioning"] = bool(enhanced)
        config["distributed"] = bool(gpu_count > 1)

        if args.dry_run:
            print(f"[dry-run] Config would be written to {config_path}")
        else:
            _dump_yaml(config, config_path)

        command = _build_command(gpu_count=gpu_count, config_path=config_path, force_torchrun=args.force_torchrun)
        if args.train_extra_args:
            command.extend(shlex.split(args.train_extra_args))

        print(f"\n=== Benchmark: {human_label} ===")
        print("Command:", " ".join(command))

        if args.dry_run:
            continue

        iteration_samples = _stream_training(command, cutoff_iters=int(args.cutoff_iters) if args.cutoff_iters else None)

        trimmed = _select_samples(iteration_samples, args.warmup_iters, args.measure_iters)
        available_after_warmup = max(0, len(iteration_samples) - args.warmup_iters)
        if args.measure_iters > available_after_warmup:
            print(
                f"[info] Early stop after {len(iteration_samples)} iterations; "
                f"only {len(trimmed)} measurement samples available.",
            )
        iter_mean, iter_std = _summarise_iteration_times(trimmed)
        eff_batch = config.get("batch_size", 0) * config.get("gradient_accumulation_steps", 0) * gpu_count
        throughput_mean, throughput_std = _summarise_throughput(trimmed, eff_batch)

        print(
            f"Collected {len(trimmed)} iterations · mean {iter_mean:.2f}s · throughput {throughput_mean:.1f} samples/s",
        )

        run_results.append(
            BenchmarkResult(
                run_name=human_label,
                gpu_count=gpu_count,
                enhanced_pxrd_conditioning=enhanced,
                batch_size=int(config.get("batch_size", 0)),
                grad_accumulation=int(config.get("gradient_accumulation_steps", 0)),
                iteration_mean_s=iter_mean,
                iteration_std_s=iter_std,
                throughput_mean=throughput_mean,
                throughput_std=throughput_std,
            )
        )

    if args.dry_run:
        print("Dry run requested; no summaries were written.")
        return

    # Sort results for consistent presentation
    run_results.sort(key=lambda item: (item.gpu_count, item.enhanced_pxrd_conditioning))
    baseline = next((result for result in run_results if not result.enhanced_pxrd_conditioning), run_results[0])
    baseline_throughput = baseline.throughput_mean
    for result in run_results:
        if math.isnan(result.throughput_mean) or baseline_throughput == 0 or math.isnan(baseline_throughput):
            result.speedup = math.nan
        else:
            result.speedup = result.throughput_mean / baseline_throughput

    _write_json(run_results, args.summary_json)
    _write_latex(run_results, args.latex_path)

    print(f"\nSaved {len(run_results)} rows to {args.latex_path}")
    print(f"JSON summary written to {args.summary_json}")


if __name__ == "__main__":
    main()
