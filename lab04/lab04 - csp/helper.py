import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from pathlib import Path
from datetime import datetime
from pprint import pprint
import json
import hashlib
import csv

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)


def random_board(n):
    """Creates a random board of size n x n. Note that only a single queen is placed in each column!"""
    
    return(np.random.randint(0,n, size = n))

def comb2(n): return n*(n-1)//2 # this is n choose 2 equivalent to math.comb(n, 2); // is int division

def conflicts(board):
    """Calculate the number of conflicts, i.e., the objective function."""

    n = len(board)
    
    horizontal_cnt = [0] * n
    diagonal1_cnt = [0] * 2 * n
    diagonal2_cnt = [0] * 2 * n
    
    for i in range(n):
        horizontal_cnt[board[i]] += 1
        diagonal1_cnt[i + board[i]] += 1
        diagonal2_cnt[i - board[i] + n] += 1
    
    return sum(map(comb2, horizontal_cnt + diagonal1_cnt + diagonal2_cnt))

# decrease the font size to fit larger boards
def show_board(board, cols = ['white', 'gray'], fontsize = 48):  
    """display the board"""
    
    n = len(board)
    
    # create chess board display
    display = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if (((i+j) % 2) != 0): 
                display[i,j] = 1
    
    cmap = colors.ListedColormap(cols)
    fig, ax = plt.subplots()
    ax.imshow(display, cmap = cmap, 
              norm = colors.BoundaryNorm(range(len(cols)+1), cmap.N))
    ax.set_xticks([])
    ax.set_yticks([])
    
    # place queens. Note: Unicode u265B is a black queen
    for j in range(n):
        plt.text(j, board[j], u"\u265B", fontsize = fontsize, 
                 horizontalalignment = 'center',
                 verticalalignment = 'center')
    
    print(f"Board with {conflicts(board)} conflicts.")
    plt.show()

def _safe_file_part(value):
    """Chuẩn hóa một giá trị để dùng an toàn trong tên file/thư mục."""

    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    ).strip("_")

    return cleaned or "experiment"


def create_experiment_id(prefix="task6"):
    """Tạo ID dùng chung cho toàn bộ các run trong một experiment."""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return f"{_safe_file_part(prefix)}_{timestamp}"


def _build_experiment_summary(runs_df):
    """Tạo bảng tổng hợp đúng định dạng yêu cầu của Task 6."""

    data = runs_df.copy()
    data["runtime_ms"] = pd.to_numeric(data["runtime_ms"], errors="coerce")
    data["best_conflicts"] = pd.to_numeric(
        data["best_conflicts"], errors="coerce"
    )
    data["iterations"] = pd.to_numeric(data["iterations"], errors="coerce")
    data["success_numeric"] = (data["best_conflicts"] == 0).astype(float)

    summary_df = (
        data.groupby(["algorithm", "board_size"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            avg_runtime_ms=("runtime_ms", "mean"),
            avg_conflicts=("best_conflicts", "mean"),
            avg_iterations=("iterations", "mean"),
            optimal_solution_rate=("success_numeric", "mean"),
        )
        .sort_values(["board_size", "algorithm"])
        .reset_index(drop=True)
    )

    summary_df["optimal_solution_rate_pct"] = (
        summary_df.pop("optimal_solution_rate") * 100
    )

    return summary_df.round(4)


def save_log_result(
    result,
    algorithm="SA",
    experiment_id=None,
    log_dir="logs",
    image_dir="img/task_5_SA",
    run_number=None,
    plot_run=True,
    show_plot=True,
    show_full_history=False,
    verbose=True,
    save_history=True,
):
    """
    Hiển thị, lưu CSV và vẽ biểu đồ kết quả của thuật toán.

    result cần có:
        board_size
        best_board
        best_conflicts
        iterations
        history
        runtime_ms
        seed

    history nên là list các dictionary:
        {
            "iteration": ...,
            "conflicts": ...,
            "best_conflicts": ...,
            "temperature": ...,
            "accepted": ...
        }

    Khi chạy batch, truyền cùng một experiment_id cho mọi run.
    Đặt plot_run=False để tránh tạo 100 biểu đồ riêng lẻ.
    save_history=False chỉ lưu metrics; mặc định lưu history CSV gzip.
    Gọi summarize_experiment() một lần khi batch hoàn tất.
    """

    required_keys = {
        "board_size",
        "best_board",
        "best_conflicts",
        "iterations",
        "history",
        "runtime_ms",
        "seed",
    }

    missing_keys = required_keys - result.keys()

    if missing_keys:
        raise ValueError(
            f"result đang thiếu các trường: {sorted(missing_keys)}"
        )

    board_size = result["board_size"]

    if not isinstance(board_size, int) or board_size <= 0:
        raise ValueError("board_size phải là một số nguyên dương.")

    if len(result["best_board"]) != board_size:
        raise ValueError(
            "board_size không khớp với số quân hậu trong best_board."
        )

    timestamp = datetime.now().astimezone()
    timestamp_text = timestamp.isoformat(timespec="seconds")
    file_timestamp = timestamp.strftime("%Y%m%d_%H%M%S_%f")

    # Nếu không truyền experiment_id, lần chạy này được cô lập thành một experiment riêng.
    if experiment_id is None:
        experiment_id = f"single_{file_timestamp}"

    experiment_id = _safe_file_part(experiment_id)
    # Hash avoids collisions between labels such as "T0=0.4" and "T0=0_4".
    algorithm_file_name = (_safe_file_part(algorithm)[:80] + "_"
                           + hashlib.sha256(algorithm.encode()).hexdigest()[:10])
    experiment_log_dir = Path(log_dir) / experiment_id
    experiment_image_dir = Path(image_dir) / experiment_id
    runs_path = experiment_log_dir / "runs.csv"
    history_dir = experiment_log_dir / "histories" / algorithm_file_name / f"n{board_size}"
    history_dir.mkdir(parents=True, exist_ok=True)

    # Stream the small index: never load historical trajectories to save a run.
    used_numbers = set()
    if runs_path.exists():
        with runs_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if "history_csv" not in (reader.fieldnames or []):
                raise ValueError("Legacy experiment: use a new experiment_id for the new log format.")
            for row in reader:
                if row["algorithm"] == algorithm and int(row["board_size"]) == board_size:
                    used_numbers.add(int(row["run_number"]))
    if run_number is None:
        run_number = max(used_numbers, default=0) + 1
    if isinstance(run_number, bool) or int(run_number) != run_number or run_number < 1:
        raise ValueError("run_number phải là số nguyên dương.")
    if run_number in used_numbers:
        raise ValueError(f"Run {run_number} của {algorithm}, n={board_size} đã tồn tại.")
    run_id = f"{algorithm_file_name}_n{board_size}_run{int(run_number):06d}"

    # Chuẩn hóa history thành DataFrame
    history = result["history"]

    if isinstance(history, pd.DataFrame):
        history_df = history.copy()
    else:
        history = list(history)

        if history and isinstance(history[0], dict):
            history_df = pd.DataFrame(history)
        else:
            # Cho phép history đơn giản dạng: [6, 5, 7, 3, 0]
            history_df = pd.DataFrame({"conflicts": history})

    # Nếu history rỗng, vẫn lưu được kết quả cuối
    if history_df.empty:
        history_df = pd.DataFrame(
            {
                "iteration": [result["iterations"]],
                "conflicts": [result["best_conflicts"]],
                "best_conflicts": [result["best_conflicts"]],
            }
        )

    if "iteration" not in history_df.columns:
        history_df.insert(0, "iteration", range(len(history_df)))

    if "conflicts" not in history_df.columns:
        raise ValueError(
            "Mỗi phần tử của history cần có trường 'conflicts'."
        )

    history_df["conflicts"] = pd.to_numeric(
        history_df["conflicts"],
        errors="coerce",
    )

    # Nếu chưa lưu best_conflicts ở từng iteration,
    # tự tính nghiệm tốt nhất tính đến thời điểm đó
    if "best_conflicts" not in history_df.columns:
        history_df["best_conflicts"] = history_df["conflicts"].cummin()

    if "temperature" not in history_df.columns:
        history_df["temperature"] = pd.NA

    if "accepted" not in history_df.columns:
        history_df["accepted"] = pd.NA

    # Only per-step values belong in history; metadata lives once in runs.csv.
    csv_path = history_dir / f"run_{int(run_number):06d}.csv.gz"
    if save_history:
        if csv_path.exists():
            raise FileExistsError(f"History already exists: {csv_path}")
        temporary_path = csv_path.with_suffix(".tmp")
        try:
            history_df.to_csv(temporary_path, index=False, compression="gzip")
            temporary_path.replace(csv_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    # runs.csv chỉ có một dòng cho mỗi lần chạy, phù hợp để tính trung bình.
    success = result["best_conflicts"] == 0
    run_record = pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "history_csv": str(csv_path.relative_to(experiment_log_dir)) if save_history else "",
                "run_id": run_id,
                "run_number": int(run_number),
                "timestamp": timestamp_text,
                "algorithm": algorithm,
                "board_size": board_size,
                "runtime_ms": result["runtime_ms"],
                "best_conflicts": result["best_conflicts"],
                "iterations": result["iterations"],
                "success": success,
                "seed": result["seed"],
                "initial_board": json.dumps(
                    result.get("initial_board"),
                    ensure_ascii=False,
                    default=_json_value,
                ),
                "metadata_json": json.dumps(result.get("metadata", {}),
                                            ensure_ascii=False, default=_json_value),
                "schedule": result.get("schedule"),
                "initial_temperature": result.get("initial_temperature"),
                "min_temperature": result.get("min_temperature"),
                "eta": result.get("eta"),
                "alpha": result.get("alpha"),
                "max_iterations": result.get("max_iterations"),
                "best_board": json.dumps(
                    result["best_board"], ensure_ascii=False, default=_json_value
                ),
            }
        ]
    )

    # Append one record; preserve the column layout of existing legacy indexes.
    if runs_path.exists():
        with runs_path.open(newline="", encoding="utf-8") as stream:
            columns = next(csv.reader(stream))
        if "history_csv" not in columns:
            raise ValueError("Legacy experiment: use a new experiment_id for the new log format.")
        run_record = run_record.reindex(columns=columns)
    run_record.to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)
    summary_path = experiment_log_dir / "summary.csv"
    # An old summary must not look current after another run is appended.
    summary_path.unlink(missing_ok=True)

    compact_result = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "run_number": int(run_number),
        "board_size": board_size,
        "best_conflicts": result["best_conflicts"],
        "runtime_ms": result["runtime_ms"],
        "success": success,
        "history_csv": str(csv_path) if save_history else None,
        "runs_csv": str(runs_path),
        "summary_csv": None,  # created by summarize_experiment after the batch
    }

    # Batch mode: vẫn lưu đầy đủ CSV nhưng bỏ qua phần vẽ tốn thời gian.
    if not plot_run:
        if verbose:
            pprint(compact_result, sort_dicts=False)
        return compact_result

    # Vẽ diễn biến conflicts
    figure, conflict_axis = plt.subplots(figsize=(10, 5))

    conflict_axis.plot(
        history_df["iteration"],
        history_df["conflicts"],
        label="Current conflicts",
        color="tab:blue",
        alpha=0.7,
    )

    conflict_axis.plot(
        history_df["iteration"],
        history_df["best_conflicts"],
        label="Best conflicts",
        color="tab:green",
        linewidth=2,
    )

    conflict_axis.set_xlabel("Iteration")
    conflict_axis.set_ylabel("Number of conflicts")
    conflict_axis.grid(alpha=0.25)

    # Vẽ temperature trên trục Y thứ hai nếu có dữ liệu
    temperature = pd.to_numeric(
        history_df["temperature"],
        errors="coerce",
    )

    axes = [conflict_axis]

    if temperature.notna().any():
        temperature_axis = conflict_axis.twinx()

        temperature_axis.plot(
            history_df["iteration"],
            temperature,
            label="Temperature",
            color="tab:red",
            linestyle="--",
            alpha=0.65,
        )

        temperature_axis.set_ylabel("Temperature")
        axes.append(temperature_axis)

    lines = []
    labels = []

    for axis in axes:
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)

    conflict_axis.legend(lines, labels, loc="upper right")
    conflict_axis.set_title(
        f"{algorithm} (n={board_size}): conflicts over iterations "
        f"(best = {result['best_conflicts']})"
    )

    figure.tight_layout()

    run_image_dir = experiment_image_dir / "runs"
    run_image_dir.mkdir(parents=True, exist_ok=True)
    image_path = run_image_dir / f"{run_id}.png"
    figure.savefig(image_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(figure)

    # In kết quả dạng dictionary
    printable_result = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "run_number": int(run_number),
        "board_size": board_size,
        "best_board": result["best_board"],
        "best_conflicts": result["best_conflicts"],
        "iterations": result["iterations"],
        "history": (
            result["history"]
            if show_full_history
            else f"{len(history_df)} records"
        ),
        "runtime_ms": result["runtime_ms"],
        "seed": result["seed"],
        "success": success,
    }

    compact_result["image_path"] = str(image_path)

    if verbose:
        pprint(printable_result, sort_dicts=False)
        print(f"\nHistory CSV: {csv_path}")
        print(f"Runs CSV: {runs_path}")
        print("Tạo summary.csv bằng summarize_experiment() sau khi batch hoàn tất.")
        print(f"Biểu đồ run: {image_path}")

    return compact_result


def summarize_experiment(
    experiment_id,
    log_dir="logs",
    image_dir="img/task_5_SA",
    show_plot=True,
    verbose=True,
):
    """Lưu bảng và biểu đồ tổng hợp sau khi chạy xong một experiment."""

    experiment_id = _safe_file_part(experiment_id)
    experiment_log_dir = Path(log_dir) / experiment_id
    experiment_image_dir = Path(image_dir) / experiment_id
    runs_path = experiment_log_dir / "runs.csv"

    if not runs_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu experiment: {runs_path}"
        )

    runs_df = pd.read_csv(runs_path)
    summary_df = _build_experiment_summary(runs_df)
    summary_path = experiment_log_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    labels = (
        summary_df["algorithm"].astype(str)
        + "\nn="
        + summary_df["board_size"].astype(str)
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("avg_runtime_ms", "Average runtime (ms)", "tab:blue"),
        ("avg_conflicts", "Average conflicts", "tab:orange"),
        ("optimal_solution_rate_pct", "Optimal runs (%)", "tab:green"),
    ]

    for axis, (column, title, color) in zip(axes, metrics):
        bars = axis.bar(labels, summary_df[column], color=color, alpha=0.8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=20)
        axis.bar_label(bars, fmt="%.2f", padding=3)

    axes[2].set_ylim(0, 105)
    figure.suptitle(f"Experiment summary: {experiment_id}")
    figure.tight_layout()

    experiment_image_dir.mkdir(parents=True, exist_ok=True)
    image_path = experiment_image_dir / "summary.png"
    figure.savefig(image_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(figure)

    if verbose:
        print(f"Runs CSV: {runs_path}")
        print(f"Summary CSV: {summary_path}")
        print(f"Summary chart: {image_path}")

    return summary_df

def _json_value(value):
    """Keep NumPy boards as JSON arrays, rather than opaque strings."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_run_history(experiment_id, run_id, log_dir="logs", chunksize=None):
    """Read only one run; chunksize returns an iterator for large histories."""
    directory = Path(log_dir) / _safe_file_part(experiment_id)
    with (directory / "runs.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["run_id"] == run_id:
                relative = row.get("history_csv")
                if not relative:
                    raise FileNotFoundError("Run has no indexed history (disabled or legacy format).")
                return pd.read_csv(directory / relative, chunksize=chunksize)
    raise KeyError(f"Unknown run_id: {run_id}")
