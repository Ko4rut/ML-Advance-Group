"""Thực nghiệm Task 6: dùng lại log SA, lưu cache HC và xuất bảng/hình."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from helper import save_log_result, _build_experiment_summary

CACHE_VERSION = 2
SA_ALGORITHM = "Simulated Annealing - Exponential T0=0.4"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def resolve_task5_source(source=None, log_dir="logs"):
    """Chọn lần chạy hoàn tất mới nhất theo timestamp; cho phép chỉ định nguồn."""
    def complete(path):
        # experiment.json được Task 5 ghi cuối cùng, sau khi hoàn tất mọi run.
        return all((path / name).is_file() for name in
                   ("experiment.json", "runs.csv", "schedule_runs.csv", "schedule_summary.csv"))

    if source is not None:
        path = Path(source)
        if not complete(path):
            raise FileNotFoundError(f"Log Task 5 chưa đầy đủ: {path}. Hãy chạy xong Task 5.")
        return path
    candidates = sorted(
        (path for path in Path(log_dir).glob("task5_SA_schedules_*")
         if re.fullmatch(r"task5_SA_schedules_\d{8}_\d{6}_\d{6}", path.name)
         and complete(path)),
        key=lambda path: path.name, reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Không có log Task 5 hoàn tất trong {log_dir}. Hãy chạy Task 5 trước.")
    return candidates[0]


def _load_scalability(source):
    """Tạo các run SA còn thiếu; lần sau dùng lại đúng cache của nguồn Task 5."""
    from simulated_annealing import simulated_annealing
    from helper import save_log_result

    folder = source.parent / (source.name + "_task6_scalability_v2")
    runs_path = folder / "runs.csv"
    existing = pd.read_csv(runs_path) if runs_path.is_file() else pd.DataFrame()
    done = set(zip(existing.board_size, existing.run_number)) if not existing.empty else set()
    for size in (12, 16, 20):
        for run in range(1, 101):
            if (size, run) in done:
                continue
            seed = 10000 + size * 100 + run
            board = np.random.RandomState(seed).randint(size, size=size)
            result = simulated_annealing(
                n=size, board=board, seed=seed, mode=1,
                initial_temperature=0.4, max_iterations=1000,
            )
            save_log_result(
                result, algorithm="Simulated Annealing - Exponential T0=0.4",
                experiment_id=folder.name, log_dir=source.parent,
                run_number=run, plot_run=False, show_plot=False, verbose=False,
            )
    runs = pd.read_csv(runs_path)
    _build_experiment_summary(runs).to_csv(folder / "summary.csv", index=False)
    (folder / "experiment.json").write_text(json.dumps({
        "source_sa": str(source), "cache_version": CACHE_VERSION,
        "algorithm": SA_ALGORITHM, "board_sizes": [12, 16, 20],
        "runs_per_size": 100, "max_iterations": 1000,
        "seed_rule": "10000 + size * 100 + run",
    }, indent=2), encoding="utf-8")
    return runs


def run_analysis(algorithms, restart_algorithm, source=None):
    """100 run/cấu hình/kích thước; cache từng run, không đo chi phí ghi file."""
    source = resolve_task5_source(source)
    output = source / "task6_all_algorithms_v2"
    images = Path("img/task_6_all") / source.name / "v2"
    output.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    cache = _read_cache_index(output)
    original = pd.read_csv(source / "runs.csv")
    original = original.loc[original.algorithm.eq("Simulated Annealing - Exponential T0=0.4")].copy()
    extra = _load_scalability(source)
    sa = pd.concat([original, extra], ignore_index=True)
    assert sa.groupby("board_size").size().to_dict() == {n: 100 for n in (4, 8, 12, 16, 20)}
    sa["algorithm"] = "Simulated Annealing"
    records = sa[["algorithm", "board_size", "run_number", "runtime_ms", "best_conflicts", "iterations", "initial_board"]].to_dict("records")
    examples = {}
    sa_example = original.loc[(original.board_size == 8) & (original.run_number == 1)].iloc[0]
    examples["Simulated Annealing"] = {
        "history": _read_source_history(source, sa_example).conflicts.tolist(),
        "unit": "Proposed moves",
    }
    saved_rng = np.random.get_state()
    try:
        for name, algorithm in algorithms.items():
            for size in (4, 8, 12, 16, 20):
                starts = sa.loc[sa.board_size.eq(size)].sort_values("run_number")
                for row in starts.itertuples():
                    board = np.array(json.loads(row.initial_board), dtype=int)
                    result = _cached_run(output, name, size, row.run_number, board, algorithm, cache)
                    records.append(_record(name, size, row.run_number, board, result))
                    if size == 8 and row.run_number == 1:
                        examples[name] = {"history": result.get("proposal_history", result["history"]),
                                          "unit": "Proposed moves" if "proposal_history" in result else "Accepted improving moves"}
            # Random restart được báo cáo riêng vì ngân sách khác single run.
            rr_name = "Restart " + name
            for size in (4, 8):
                starts = sa.loc[sa.board_size.eq(size)].sort_values("run_number")
                for row in starts.itertuples():
                    board = np.array(json.loads(row.initial_board), dtype=int)
                    def with_restart(n, initial_board):
                        return restart_algorithm(algorithm, n, max_restarts=100, initial_board=initial_board)
                    result = _cached_run(output, rr_name, size, row.run_number, board, with_restart, cache)
                    records.append(_record(rr_name, size, row.run_number, board, result))
                    if size == 8 and row.run_number == 1:
                        examples[rr_name] = {"history": result["full_history"], "boundaries": result["restart_boundaries"],
                                             "unit": "Recorded search steps (including restart starts)"}
    finally:
        np.random.set_state(saved_rng)
    runs = pd.DataFrame(records)
    assert not runs.duplicated(["algorithm", "board_size", "run_number"]).any()
    assert runs.groupby(["algorithm", "board_size"]).size().eq(100).all()
    # runs.csv remains the append-only HC index, used for resuming interrupted batches.
    runs.to_csv(output / "analysis_runs.csv", index=False)
    summary = runs.groupby(["algorithm", "board_size"], sort=False).agg(
        runs=("runtime_ms", "size"), avg_runtime_ms=("runtime_ms", "mean"),
        avg_conflicts=("best_conflicts", "mean"),
        success_pct=("best_conflicts", lambda scores: 100 * scores.eq(0).mean()),
    ).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    (output / "experiment.json").write_text(json.dumps({
        "source_sa": str(source), "sa_config": "Exponential T0=0.4", "sa_max_iterations": 1000,
        "runs_per_size": 100, "hc_seed_rule": "30000 + size*100 + run_number",
        "restart_budget": "1 initial run + at most 100 restarts",
        "cache_version": CACHE_VERSION, "note": "Delete or rename cache directory if algorithm/settings change.",
    }, indent=2), encoding="utf-8")
    return runs, summary, examples, output, images


def _read_source_history(source, row):
    """Use the new index; still allow reading an untouched legacy Task 5 source."""
    relative = getattr(row, "history_csv", None)
    if isinstance(relative, str) and relative:
        return pd.read_csv(source / relative)
    paths = list((source / "histories").rglob(f"{row.run_id}.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Không tìm thấy history của run {row.run_id} trong {source}")
    return pd.read_csv(paths[0])


def _read_cache_index(output):
    path = output / "runs.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path).to_dict("records")
    cache = {}
    for row in rows:
        key = (row["algorithm"], int(row["board_size"]), int(row["run_number"]))
        if key in cache:
            raise ValueError(f"Duplicate cached run: {key}")
        cache[key] = row
    return cache


def _cached_run(output, name, size, run, board, algorithm, cache=None):
    """Persist metrics once and one compressed trajectory, including restart markers."""
    cache = _read_cache_index(output) if cache is None else cache
    key = (name, size, run)
    if key in cache:
        row = cache[key]
        if json.loads(row["initial_board"]) != board.tolist():
            raise ValueError(f"Starting board differs from cache: {name}, n={size}, run={run}")
        metadata = json.loads(row["metadata_json"])
        # Only run 1 at n=8 needs a trajectory for the convergence plots.
        history = []
        if size == 8 and run == 1:
            history = pd.read_csv(output / row["history_csv"]).conflicts.tolist()
        elif not (output / row["history_csv"]).is_file():
            raise FileNotFoundError(output / row["history_csv"])
        result = {
            "board": json.loads(row["best_board"]), "conflicts": row["best_conflicts"],
            "time_ms": row["runtime_ms"], "iterations": row["iterations"],
            "history": history, **metadata,
        }
        if metadata["history_kind"] != "history":
            result[metadata["history_kind"]] = history
        return result
    seed = 30000 + size * 100 + run
    np.random.seed(seed)
    result = algorithm(size, initial_board=board.copy())
    history_kind = next(key for key in ("full_history", "proposal_history", "history") if key in result)
    metadata = {key: value for key, value in result.items()
                if key not in {"board", "conflicts", "time_ms", "iterations", "history",
                               "full_history", "proposal_history", "restart_history"}}
    metadata["history_kind"] = history_kind
    normalized = {
        "board_size": int(size), "initial_board": board, "best_board": result["board"],
        "best_conflicts": result["conflicts"], "runtime_ms": result["time_ms"],
        "iterations": result["iterations"], "seed": seed,
        "history": result[history_kind], "metadata": metadata,
    }
    saved = save_log_result(normalized, algorithm=name, experiment_id=output.name,
                           log_dir=output.parent, run_number=int(run),
                           plot_run=False, show_plot=False, verbose=False)
    cache[key] = {
        "initial_board": json.dumps(board.tolist()),
        "best_board": json.dumps(result["board"], default=_json_default),
        "best_conflicts": result["conflicts"], "runtime_ms": result["time_ms"],
        "iterations": result["iterations"], "metadata_json": json.dumps(metadata, default=_json_default),
        "history_csv": str(Path(saved["history_csv"]).relative_to(output)),
    }
    return result


def _record(name, size, run, board, result):
    return {"algorithm": name, "board_size": size, "run_number": run,
            "runtime_ms": result["time_ms"], "best_conflicts": result["conflicts"],
            "iterations": result["iterations"], "initial_board": json.dumps(board.tolist()),
            "restarts": result.get("restarts", 0)}


def plot_convergence(examples, images):
    """Mỗi trục ghi đúng đơn vị; run 1 được chọn trước khi xem kết quả."""
    core = [name for name in examples if not name.startswith("Restart ")]
    restart = [name for name in examples if name.startswith("Restart ")]
    for names, shape, filename in [(core, (2, 2), "convergence_core.png"),
                                    (restart, (1, 3), "convergence_restarts.png")]:
        fig, axes = plt.subplots(*shape, figsize=(12, 6 if shape[0] == 2 else 3.8))
        for ax, name in zip(np.asarray(axes).flat, names):
            example = examples[name]
            scores = np.asarray(example["history"])
            ax.plot(scores, label="Current conflicts", alpha=0.75)
            ax.plot(np.minimum.accumulate(scores), "--", label="Best so far")
            for boundary in example.get("boundaries", [])[1:]:
                ax.axvline(boundary, color="gray", linestyle=":", alpha=0.5)
            ax.set(title=name, xlabel=example["unit"], ylabel="Conflicting pairs", ylim=(0, max(scores)+1))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle("8 queens: paired run 1")
        fig.tight_layout()
        fig.savefig(images / filename, dpi=150, bbox_inches="tight")
        plt.show()


def plot_scaling(summary, output, images):
    core = summary.loc[~summary.algorithm.str.startswith("Restart ")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fits = []
    for name, group in core.groupby("algorithm", sort=False):
        group = group.sort_values("board_size")
        x, y = group.board_size.to_numpy(), group.avg_runtime_ms.to_numpy()
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        predicted = intercept + slope * np.log(x)
        r2 = 1 - np.sum((np.log(y)-predicted)**2) / np.sum((np.log(y)-np.log(y).mean())**2)
        fits.append({"algorithm": name, "empirical_exponent": slope, "r_squared": r2})
        line, = axes[0].loglog(x, y, "o-", label=f"{name}: p={slope:.2f}")
        axes[0].loglog(x, np.exp(predicted), "--", color=line.get_color(), alpha=0.5)
        axes[1].plot(x, group.success_pct, "o-", label=name)
    axes[0].set(xlabel="Board size n", ylabel="Mean runtime (ms)", title="Runtime scaling (log-log)")
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set(xlabel="Board size n", ylabel="Success rate (%)", ylim=(-5, 105), title="Solution quality")
    for ax in axes:
        ax.set_xticks([4, 8, 12, 16, 20], [4, 8, 12, 16, 20])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(images / "scalability_all.png", dpi=150, bbox_inches="tight")
    plt.show()
    fits = pd.DataFrame(fits)
    fits.to_csv(output / "scalability_fit.csv", index=False)
    return fits
