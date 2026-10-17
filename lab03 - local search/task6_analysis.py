"""Thực nghiệm Task 6: dùng lại log SA, lưu cache HC và xuất bảng/hình."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run_analysis(algorithms, restart_algorithm, source):
    """100 run/cấu hình/kích thước; cache từng run, không đo chi phí ghi file."""
    source = Path(source)
    output = source / "task6_all_algorithms_v1"
    images = Path("img/task_6_all") / source.name
    output.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(source / "runs.csv")
    original = original.loc[original.algorithm.eq("Simulated Annealing - Exponential T0=0.4")].copy()
    extra = pd.read_csv(source.parent / (source.name + "_task6_scalability") / "runs.csv")
    sa = pd.concat([original, extra], ignore_index=True)
    assert sa.groupby("board_size").size().to_dict() == {n: 100 for n in (4, 8, 12, 16, 20)}
    sa["algorithm"] = "Simulated Annealing"
    records = sa[["algorithm", "board_size", "run_number", "runtime_ms", "best_conflicts", "iterations", "initial_board"]].to_dict("records")
    examples = {}
    sa_example = original.loc[(original.board_size == 8) & (original.run_number == 1)].iloc[0]
    path, = (source / "histories").rglob(f"{sa_example.run_id}.csv")
    examples["Simulated Annealing"] = {"history": pd.read_csv(path).conflicts.tolist(), "unit": "Proposed moves"}
    saved_rng = np.random.get_state()
    try:
        for name, algorithm in algorithms.items():
            for size in (4, 8, 12, 16, 20):
                starts = sa.loc[sa.board_size.eq(size)].sort_values("run_number")
                for row in starts.itertuples():
                    board = np.array(json.loads(row.initial_board), dtype=int)
                    result = _cached_run(output, name, size, row.run_number, board, algorithm)
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
                    result = _cached_run(output, rr_name, size, row.run_number, board, with_restart)
                    records.append(_record(rr_name, size, row.run_number, board, result))
                    if size == 8 and row.run_number == 1:
                        examples[rr_name] = {"history": result["full_history"], "boundaries": result["restart_boundaries"],
                                             "unit": "Recorded search steps (including restart starts)"}
    finally:
        np.random.set_state(saved_rng)
    runs = pd.DataFrame(records)
    assert not runs.duplicated(["algorithm", "board_size", "run_number"]).any()
    assert runs.groupby(["algorithm", "board_size"]).size().eq(100).all()
    runs.to_csv(output / "runs.csv", index=False)
    summary = runs.groupby(["algorithm", "board_size"], sort=False).agg(
        runs=("runtime_ms", "size"), avg_runtime_ms=("runtime_ms", "mean"),
        avg_conflicts=("best_conflicts", "mean"),
        success_pct=("best_conflicts", lambda scores: 100 * scores.eq(0).mean()),
    ).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    (output / "examples.json").write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    (output / "experiment.json").write_text(json.dumps({
        "source_sa": str(source), "sa_config": "Exponential T0=0.4", "sa_max_iterations": 1000,
        "runs_per_size": 100, "hc_seed_rule": "30000 + size*100 + run_number",
        "restart_budget": "1 initial run + at most 100 restarts",
        "cache_version": 1, "note": "Delete or rename cache directory if algorithm/settings change.",
    }, indent=2), encoding="utf-8")
    return runs, summary, examples, output, images


def _cached_run(output, name, size, run, board, algorithm):
    folder = output / "hc_runs" / name.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"n{size}_run{run:03d}.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if stored["initial_board"] != board.tolist():
            raise ValueError(f"Starting board differs from cache: {path}")
        return stored["result"]
    seed = 30000 + size * 100 + run
    np.random.seed(seed)
    result = algorithm(size, initial_board=board)
    payload = {"initial_board": board.tolist(), "seed": seed, "result": result}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
    temporary.replace(path)
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
