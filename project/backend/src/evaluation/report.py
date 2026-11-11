"""Regenerate ``summary/`` (CSV, plots, Vietnamese ``report.md``) from a run directory.

Inputs: ``manifest.json`` and ``games.jsonl`` only. Nothing is read from earlier
summaries, and CSV/report text carries no generation timestamp, so running it
twice gives byte-identical CSV files and ``report.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from . import metrics
from .metrics import MetricsCollector, fmt_num

LOGGER = logging.getLogger("tournament.report")
SUMMARY_DIRNAME = "summary"
FALLBACK_WARN_RATE = 0.5

CONCLUSION_TEXT = {
    "A_stronger": "{A} mạnh hơn",
    "B_stronger": "{B} mạnh hơn",
    "inconclusive": "chưa đủ bằng chứng",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"not a tournament run directory (no manifest.json): {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_games(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read ``games.jsonl`` without modifying it; returns (records by game_index, notes).

    A torn final line (crash) is skipped with a note; a corrupt middle line or a
    duplicate ``game_index`` is an error.
    """
    path = run_dir / "games.jsonl"
    notes: list[str] = []
    if not path.is_file():
        return [], ["games.jsonl chưa tồn tại"]
    pieces = path.read_bytes().split(b"\n")
    tail = pieces.pop()
    lines = [(i, piece) for i, piece in enumerate(pieces, start=1) if piece.strip()]
    if tail.strip():
        lines.append((len(pieces) + 1, tail))
    records: dict[int, dict[str, Any]] = {}
    for position, (number, piece) in enumerate(lines):
        try:
            record = json.loads(piece)
        except (ValueError, UnicodeDecodeError):
            if position == len(lines) - 1:
                notes.append(f"dòng {number} của games.jsonl bị cắt dở nên đã bỏ qua")
                continue
            raise ValueError(f"games.jsonl: corrupt record on line {number}") from None
        index = record["game_index"]
        if index in records:
            raise ValueError(f"games.jsonl: duplicate game_index {index} (line {number})")
        records[index] = record
    return [records[i] for i in sorted(records)], notes


# --------------------------------------------------------------------------
# Plots (optional)
# --------------------------------------------------------------------------


def _load_pyplot():
    """Return matplotlib.pyplot with the Agg backend, or ``None`` if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - ImportError or broken install: plots are optional
        return None
    return plt


def make_plots(summary: Mapping[str, Any], directory: Path) -> list[str]:
    """Write the three PNGs; returns their file names ([] if matplotlib is missing)."""
    plt = _load_pyplot()
    if plt is None:
        LOGGER.warning("matplotlib not available: skipping plots")
        print("WARNING: matplotlib chưa được cài, bỏ qua biểu đồ (pip install matplotlib).")
        return []
    directory.mkdir(parents=True, exist_ok=True)
    agents = summary["agents"]
    names: list[str] = []

    # 1. crosstable heatmap
    n = len(agents)
    matrix = [[row.get(b) if a != b else None for b in agents] for a, row in
              ((r["agent"], r) for r in summary["crosstable"])]
    fig, ax = plt.subplots(figsize=(1.1 * n + 3, 1.0 * n + 2.2))
    shown = [[float("nan") if v is None else v for v in row] for row in matrix]
    image = ax.imshow(shown, cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(n), agents, rotation=30, ha="right")
    ax.set_yticks(range(n), agents)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] is not None:
                ax.text(j, i, f"{matrix[i][j]:.0f}", ha="center", va="center", fontsize=10)
    ax.set_xlabel("đối thủ")
    ax.set_ylabel("agent (điểm % của hàng khi gặp cột)")
    ax.set_title("Bảng chéo: % điểm")
    fig.colorbar(image, ax=ax, label="% điểm")
    fig.tight_layout()
    fig.savefig(directory / "crosstable_heatmap.png", dpi=130)
    plt.close(fig)
    names.append("crosstable_heatmap.png")

    # 2. Elo with CI
    rows = sorted(summary["standings"], key=lambda r: (r["elo"] is None, -(r["elo"] or 0)))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1.8))
    ys = list(range(len(rows)))
    elos = [r["elo"] or 0.0 for r in rows]
    lows = [(r["elo"] or 0.0) - (r["elo_low"] if r["elo_low"] is not None else r["elo"] or 0.0) for r in rows]
    highs = [(r["elo_high"] if r["elo_high"] is not None else r["elo"] or 0.0) - (r["elo"] or 0.0) for r in rows]
    ax.errorbar(elos, ys, xerr=[lows, highs], fmt="o", capsize=4, color="#1f4e79")
    ax.set_yticks(ys, [r["agent"] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Elo (neo {summary['anchor']} = 0), khoảng tin cậy bootstrap 95%")
    ax.set_title("Elo theo Bradley–Terry")
    fig.tight_layout()
    fig.savefig(directory / "elo_ci.png", dpi=130)
    plt.close(fig)
    names.append("elo_ci.png")

    # 3. cost per move
    cost = [r for r in summary["cost"] if r["moves"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if cost:
        xs = range(len(cost))
        ax.bar(xs, [max(r["think_mean_s"], 1e-6) for r in cost], color="#8fb3d9", label="trung bình")
        ax.scatter(xs, [max(r["think_p95_s"], 1e-6) for r in cost], color="#c0392b", zorder=3, label="p95")
        ax.set_xticks(list(xs), [r["agent"] for r in cost])
        ax.set_yscale("log")
        ax.legend()
    ax.set_ylabel("giây suy nghĩ mỗi nước (thang log)")
    ax.set_title("Chi phí tính toán mỗi nước")
    fig.tight_layout()
    fig.savefig(directory / "cost_per_move.png", dpi=130)
    plt.close(fig)
    names.append("cost_per_move.png")
    return names


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} giờ {minutes:02d} phút"
    if minutes:
        return f"{minutes} phút {secs:02d} giây"
    return f"{secs} giây"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(out)


def _pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else fmt_num(value, digits) + "%"


def _num(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else fmt_num(value, digits)


def _params_text(info: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in info.get("params", {}).items():
        if key == "checkpoint":
            continue
        if key == "checkpoint_sha256":
            parts.append(f"checkpoint sha256={str(value)[:12]}…")
        elif key == "config" and isinstance(value, dict):
            parts.append("config(" + ", ".join(f"{k}={v}" for k, v in value.items()) + ")")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) or "mặc định"


def _budget_text(info: Mapping[str, Any]) -> str:
    budget = info.get("budget")
    if not budget:
        return "không có budget riêng"
    return ", ".join(f"{k}={v}" for k, v in budget.items())


def _pair_label(row: Mapping[str, Any]) -> str:
    return f"{row['A']} vs {row['B']}"


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_warnings(
    manifest: Mapping[str, Any], summary: Mapping[str, Any], completed: int, total: int,
    load_notes: list[str],
) -> list[str]:
    warnings: list[str] = []
    totals = summary["totals"]
    cost = {r["agent"]: r for r in summary["cost"]}
    agents_info = manifest.get("agents", {})

    if total and completed < total:
        warnings.append(f"**Chưa hoàn tất:** mới có {completed}/{total} ván; mọi số liệu chỉ là tạm thời.")
    warnings.extend(f"**Log:** {note}." for note in load_notes)
    if totals["error"]:
        warnings.append(
            f"**Có {totals['error']} ván lỗi (error)**: các ván này bị loại khỏi mọi phép tính điểm, "
            "xem `errors.csv`. Tổng số ván hợp lệ của từng cặp vì thế có thể dưới số ván dự kiến.")
    if totals["truncated"]:
        warnings.append(
            f"**{totals['truncated']} ván truncated** (đạt giới hạn nước đi, chưa phân thắng thua): "
            "không phải hòa. Bảng chính tính chúng 0.5 điểm (`T_half`), đồng thời có cách tính loại chúng ra (`excl_T`).")
    if summary["rank_differs"]:
        a = ", ".join(summary["rank_differs"])
        warnings.append(
            f"**Thứ hạng đổi theo cách tính truncated** (agent: {a}): xếp hạng theo `T_half` khác xếp hạng "
            "theo `excl_T`, nên kết luận về các agent này nhạy với quy ước truncated.")

    low_distinct = [r for r in summary["pairs"] if r["distinct_ratio"] is not None and r["distinct_ratio"] < 1]
    if low_distinct:
        listing = "; ".join(
            f"{_pair_label(r)} ({r['distinct_games']}/{r['_valid']} ván khác nhau)" for r in low_distinct)
        warnings.append(
            "**Có ván trùng nhau hoàn toàn** (`distinct_ratio` < 1): " + listing + ". "
            "Các ván trùng không mang thêm thông tin độc lập, nên khoảng tin cậy có thể hẹp hơn thực tế. "
            "Nguyên nhân thường gặp: hai agent tất định (hoặc một agent gần như trùng agent kia, như tabular_q khi hầu hết nước rơi về `rules`) "
            "đổi màu cho nhau vẫn đi ra cùng một ván.")

    q = cost.get("tabular_q")
    if q and q["fallback_rate"] is not None:
        rate = q["fallback_rate"] * 100
        if q["fallback_rate"] >= FALLBACK_WARN_RATE:
            warnings.append(
                f"**tabular_q gần như là `rules`:** {fmt_num(rate, 1)}% số nước của nó rơi về luật cơ bản "
                "(trạng thái không có trong bảng Q). Kết quả của tabular_q chủ yếu phản ánh `rules`, "
                "không phải năng lực của bảng Q đã học.")
        else:
            warnings.append(
                f"**tabular_q:** {fmt_num(rate, 1)}% số nước dùng fallback `rules` (phần còn lại lấy từ bảng Q).")

    nondeterministic = [n for n, info in agents_info.items() if not info.get("is_deterministic", True)]
    if nondeterministic:
        warnings.append(
            "**Agent không tất định:** " + ", ".join(f"`{n}`" for n in nondeterministic) +
            " không tái lập chính xác từng nước khi chạy lại (chỉ tái lập về mặt thống kê).")
    if "random" in agents_info and not agents_info["random"].get("is_deterministic", True):
        warnings.append(
            "`random` có seed riêng cho mỗi ván nên về nguyên tắc vẫn phát lại đúng nước, "
            "dù được đánh dấu là không tất định.")
    if "heuristic" in agents_info and not agents_info["heuristic"].get("is_deterministic", True):
        limit = agents_info["heuristic"].get("params", {}).get("time_limit")
        extra = ""
        h = cost.get("heuristic")
        if h and h["think_p95_s"] is not None and limit:
            extra = f" (p95 thời gian nghĩ thực đo: {fmt_num(h['think_p95_s'], 3)} giây"
            if h["think_median_s"] is not None and h["think_median_s"] >= 0.9 * float(limit):
                extra += f"; trung vị {fmt_num(h['think_median_s'], 3)} giây, tức phần lớn nước đi dừng vì hết giờ chứ không phải vì hết node"
            extra += ")"
        warnings.append(
            f"**heuristic phụ thuộc đồng hồ:** agent này có giới hạn thời gian {limit} giây mỗi nước nên "
            f"nước đi có thể khác nhau tùy tốc độ và độ bận của máy{extra}.")
    if "mcts" in agents_info:
        m = cost.get("mcts")
        sims = _num(m["simulations_mean"], 0) if m else "–"
        warnings.append(
            f"**mcts dùng thời gian:** mỗi nước chỉ được nghĩ trong thời gian cố định nên số mô phỏng trung bình là "
            f"{sims} mỗi nước. Chạy nhiều worker song song (lần chạy này: {manifest.get('workers', '?')} worker) làm máy "
            "bận hơn, số mô phỏng giảm và mcts yếu đi; chạy lại với số worker khác có thể cho kết quả hơi khác. "
            "Muốn tái lập chính xác, đặt `max_simulations` trong config.")

    git = (manifest.get("provenance") or {}).get("git") or {}
    if git.get("dirty"):
        warnings.append(
            f"**Cây git bẩn** lúc chạy ({len(git.get('status', []))} thay đổi chưa commit): kết quả không gắn với "
            "một commit sạch. Xem `git.diff` trong thư mục run.")
    if manifest.get("excluded_agents"):
        listing = "; ".join(f"`{n}`: {why}" for n, why in manifest["excluded_agents"].items())
        warnings.append(f"**Agent bị loại khỏi giải:** {listing}.")
    history = manifest.get("resume_history") or []
    if history:
        changed = [h for h in history if h.get("code_changed")]
        text = f"**Run đã resume {len(history)} lần.**"
        if changed:
            text += (f" Trong đó {len(changed)} lần commit git đã đổi so với lúc bắt đầu "
                     "(`--allow-code-change`): các ván của cùng một run có thể được chơi bằng mã khác nhau.")
        warnings.append(text)
    if manifest.get("status") == "failed":
        warnings.append("**Run kết thúc với trạng thái `failed`** theo manifest.")
    return warnings


def build_report(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    run_dir: Path,
    completed: int,
    plots: list[str],
    load_notes: list[str],
) -> str:
    total = manifest.get("total_games", completed)
    totals = summary["totals"]
    agents = summary["agents"]
    run_id = manifest.get("run_id", run_dir.name)
    prov = manifest.get("provenance", {}) or {}
    git = prov.get("git", {}) or {}
    system = prov.get("system", {}) or {}
    warnings = build_warnings(manifest, summary, completed, total, load_notes)
    lines: list[str] = []
    add = lines.append

    add(f"# Báo cáo giải đấu vòng tròn: {manifest.get('name', run_id)}")
    add("")
    if completed < total:
        add(f"> ⚠️ **Chưa hoàn tất: {completed}/{total} ván.** Số liệu dưới đây chỉ là tạm thời.")
        add("")
    if totals["error"]:
        add(f"> ⚠️ Có **{totals['error']} ván lỗi** bị loại khỏi tính điểm (xem mục 8 và `errors.csv`).")
        add("")

    # 1 ---------------------------------------------------------------
    add("## 1. Thông tin lần chạy")
    add("")
    commit = (git.get("commit") or "không rõ")[:10]
    dirty = "có (cây git bẩn)" if git.get("dirty") else ("không" if git.get("dirty") is not None else "không rõ")
    info_rows = [
        ["Run ID", f"`{run_id}`"],
        ["Chế độ", manifest.get("mode", "full")],
        ["Commit / nhánh", f"`{commit}` / `{git.get('branch') or 'không rõ'}`"],
        ["Thay đổi chưa commit", dirty],
        ["Máy", f"{system.get('platform', '?')}, {system.get('cpu_count', '?')} luồng CPU"],
        ["Python", f"{system.get('python_implementation', 'Python')} {system.get('python', '?')}"],
        ["Config hash", f"`{str(manifest.get('config_hash', ''))[:16]}`"],
        ["Số worker", manifest.get("workers", "?")],
        ["Bắt đầu / kết thúc (UTC)", f"{manifest.get('started_at', '?')} / {manifest.get('finished_at') or '?'}"],
        ["Tổng thời gian chạy", _duration(manifest.get("total_seconds"))],
        ["Ván hoàn tất", f"{completed}/{total}"],
        ["Ván kết thúc bình thường", totals["finished"]],
        ["Ván hòa", totals["draws"]],
        ["Ván truncated", totals["truncated"]],
        ["Ván lỗi (error)", totals["error"]],
    ]
    add(_table(["Mục", "Giá trị"], info_rows))
    add("")

    # 2 ---------------------------------------------------------------
    add("## 2. Thiết kế thực nghiệm")
    add("")
    config = manifest.get("config", {}) or {}
    openings = config.get("openings", [])
    gpp = manifest.get("games_per_pair", "?")
    colors = config.get("colors_per_opening", 2)
    add(f"- **Đấu vòng tròn:** {len(agents)} agent, mỗi cặp {gpp} ván "
        f"({len(openings)} thế khai cuộc × {colors} màu), tổng {total} ván.")
    add(f"- **Vì sao cần khai cuộc cố định:** các agent tìm kiếm (`rules`, `alphabeta`, `tabular_q`) là tất định. "
        f"Nếu cho hai agent tất định đánh nhiều lần từ thế ban đầu thì mọi ván giống hệt nhau và {gpp} ván chỉ là 1 ván "
        f"lặp lại {gpp} lần. Sự đa dạng vì vậy phải đến từ thế khai cuộc: mỗi ván bắt đầu từ một khai cuộc cố định "
        f"(bàn cờ cuối khác nhau, sách `{config.get('opening_book_version', '?')}`).")
    add("- **Cân bằng màu:** mỗi thế được đánh hai lần trong mỗi cặp, A cầm Đỏ một ván và cầm Đen một ván, "
        "để lợi thế đi trước và thế khai cuộc lệch không thiên vị bên nào.")
    add(f"- **Giới hạn nước đi:** {config.get('max_plies', '?')} nước (tính cả nước khai cuộc). "
        "Ván chưa phân thắng thua khi chạm giới hạn được ghi là **truncated** và *không phải hòa*; "
        "bảng chính tính truncated = 0.5 điểm (`T_half`), kèm cách tính loại bỏ chúng (`excl_T`).")
    add("- **Ván lỗi (error):** agent ném lỗi hoặc đi nước không hợp lệ; ván bị loại khỏi mọi phép tính điểm.")
    add("")
    agent_rows = []
    for name in agents:
        info = (manifest.get("agents") or {}).get(name, {})
        agent_rows.append([
            f"`{name}`", info.get("class", ""), _budget_text(info), _params_text(info),
            "có" if info.get("is_deterministic") else "không",
        ])
    add(_table(["Agent", "Lớp", "Budget", "Tham số", "Tất định?"], agent_rows))
    add("")
    add("Mọi agent chạy đúng cấu hình mặc định mà web API đang dùng; giải đấu chỉ đo, không tinh chỉnh agent.")
    add("")

    # 3 ---------------------------------------------------------------
    add("## 3. Bảng xếp hạng")
    add("")
    rows = []
    for r in sorted(summary["standings"], key=lambda r: (r["rank"], agents.index(r["agent"]))):
        rows.append([
            r["rank"], f"`{r['agent']}`", r["games"], f"{r['W']}-{r['D']}-{r['L']}-{r['T']}-{r['E']}",
            _num(r["points"], 1), _pct(r["score_pct"], 2), _pct(r["score_pct_excl_T"], 2), r["rank_excl_T"],
            _num(r["elo"], 0), f"[{_num(r['elo_low'], 0)}; {_num(r['elo_high'], 0)}]",
        ])
    add(_table(["Hạng", "Agent", "Ván", "W-D-L-T-E", "Điểm", "% điểm (T=0.5)", "% điểm (loại T)",
                "Hạng (loại T)", "Elo", "Elo 95% CI"], rows))
    add("")
    add("**Cách đọc.** Thắng = 1 điểm, hòa = 0.5, thua = 0; `% điểm` là điểm trên số ván hợp lệ (không tính ván lỗi). "
        "Hạng xếp theo `% điểm (T=0.5)`.")
    add("")
    add(f"**Elo** được ước lượng theo mô hình Bradley–Terry (thuật toán MM của Hunter 2004), hòa tính bằng nửa thắng, "
        f"quy đổi `Elo = 400·log10(sức mạnh)`. Để rating không chạy ra vô cực khi một bên thắng tuyệt đối, "
        f"mỗi cặp (kể cả cặp chưa có ván) được cộng thêm **1 ván hòa ảo** (prior nhẹ). Thang Elo được neo sao cho "
        f"`{summary['anchor']}` = 0, nên chỉ có chênh lệch giữa các agent là có nghĩa. Khoảng tin cậy 95% lấy bằng bootstrap "
        f"({summary['bootstrap_samples']} lần lấy mẫu lại có hoàn lại các ván trong từng cặp, phân vị 2.5% và 97.5%); "
        "agent dùng làm neo luôn có khoảng [0; 0].")
    add("")
    if summary["rank_differs"]:
        add(f"> Lưu ý: thứ hạng theo cách tính truncated khác nhau ở các agent: {', '.join(summary['rank_differs'])}.")
    else:
        add("> Thứ hạng giống nhau dù tính truncated = 0.5 hay loại truncated.")
    add("")
    if "elo_ci.png" in plots:
        add("![Elo kèm khoảng tin cậy](plots/elo_ci.png)")
        add("")

    # 4 ---------------------------------------------------------------
    add("## 4. Bảng chéo")
    add("")
    add("Ô (hàng, cột) là % điểm của agent ở hàng khi gặp agent ở cột.")
    add("")
    cross_rows = []
    for row in summary["crosstable"]:
        cross_rows.append([f"`{row['agent']}`"] + [
            "" if row["agent"] == b else _pct(row.get(b), 1) for b in agents])
    add(_table(["", *[f"`{b}`" for b in agents]], cross_rows))
    add("")
    if "crosstable_heatmap.png" in plots:
        add("![Bảng chéo](plots/crosstable_heatmap.png)")
        add("")

    # 5 ---------------------------------------------------------------
    add("## 5. Kết quả từng cặp")
    add("")
    pair_rows = []
    for r in summary["pairs"]:
        conclusion = CONCLUSION_TEXT[r["conclusion"]].format(A=r["A"], B=r["B"])
        pair_rows.append([
            _pair_label(r), r["games"], f"{r['W']}-{r['D']}-{r['L']}-{r['T']}-{r['E']}",
            _num(r["score"], 1), _pct(r["score_pct"], 1),
            f"[{_num(r['wilson_low'], 1)}; {_num(r['wilson_high'], 1)}]",
            f"{_num(r['A_as_red_score'], 1)} / {_num(r['A_as_black_score'], 1)}",
            _num(r["avg_plies"], 1), f"{r['distinct_games']}/{r['_valid']}", conclusion,
        ])
    add(_table(["Cặp (A vs B)", "Ván", "W-D-L-T-E (của A)", "Điểm A", "% điểm A", "Wilson 95%",
                "Điểm A khi Đỏ / Đen", "Số nước TB", "Ván khác nhau", "Kết luận"], pair_rows))
    add("")
    add("**Kết luận thống kê:** nếu khoảng Wilson 95% của `% điểm A` nằm hoàn toàn trên 50% thì A mạnh hơn, "
        "hoàn toàn dưới 50% thì B mạnh hơn, còn lại là *chưa đủ bằng chứng*. Khoảng Wilson chỉ là **xấp xỉ** khi có hòa "
        "(hòa được tính 0.5 \"thành công\", không phải phép thử nhị phân thuần túy). Ngoài ra có nhiều cặp được so sánh "
        "cùng lúc mà không hiệu chỉnh cho so sánh nhiều lần, nên cần thận trọng với các cặp sát ngưỡng.")
    add("")

    # 6 ---------------------------------------------------------------
    add("## 6. Chi phí tính toán")
    add("")
    cost_rows = []
    for r in summary["cost"]:
        cost_rows.append([
            f"`{r['agent']}`", r["moves"], _num(r["think_mean_s"], 4), _num(r["think_median_s"], 4),
            _num(r["think_p95_s"], 4), _num(r["think_total_s"], 1), _num(r["nodes_mean"], 1),
            _num(r["simulations_mean"], 1),
            "–" if r["fallback_rate"] is None else _pct(100 * r["fallback_rate"], 1),
        ])
    add(_table(["Agent", "Số nước", "TB (giây)", "Trung vị (giây)", "p95 (giây)", "Tổng (giây)",
                "Nodes TB", "Mô phỏng TB", "Tỉ lệ fallback"], cost_rows))
    add("")
    add("Chỉ tính các nước do agent tự đi (không tính nước khai cuộc). Thời gian là thời gian suy nghĩ đo trong "
        "tiến trình worker, nên phụ thuộc máy và số worker chạy song song.")
    add("")
    if "cost_per_move.png" in plots:
        add("![Chi phí mỗi nước](plots/cost_per_move.png)")
        add("")

    # 7 ---------------------------------------------------------------
    add("## 7. Phân tích màu và khai cuộc")
    add("")
    color_rows = []
    for r in summary["color"]:
        label = "**Toàn giải (lợi thế đi trước)**" if r["agent"] == "TOTAL" else f"`{r['agent']}`"
        color_rows.append([
            label, r["red_games"], _pct(r["red_score_pct"], 1), r["black_games"], _pct(r["black_score_pct"], 1),
            _num(r["red_minus_black_pct"], 1),
        ])
    add(_table(["Agent", "Ván cầm Đỏ", "% điểm khi Đỏ", "Ván cầm Đen", "% điểm khi Đen", "Đỏ − Đen (điểm %)"],
               color_rows))
    add("")
    total_row = next((r for r in summary["color"] if r["agent"] == "TOTAL"), None)
    if total_row and total_row["red_score_pct"] is not None:
        add(f"Trên toàn giải, bên Đỏ (đi trước) đạt {_pct(total_row['red_score_pct'], 1)} số điểm.")
        add("")
    open_rows = []
    for r in summary["openings"]:
        open_rows.append([
            f"`{r['opening_id']}`", r["family"], r["games"], _pct(r["red_win_pct"], 0), _pct(r["draw_pct"], 0),
            _pct(r["black_win_pct"], 0), _pct(r["truncated_pct"], 0), _pct(r["error_pct"], 0),
            {"red": "⚠️ lệch Đỏ", "black": "⚠️ lệch Đen"}.get(r["skew"], ""),
        ])
    add(_table(["Thế khai cuộc", "Họ", "Ván", "Đỏ thắng", "Hòa", "Đen thắng", "Truncated", "Lỗi", "Ghi chú"], open_rows))
    add("")
    skewed = [r for r in summary["openings"] if r["skew"]]
    if skewed:
        add(f"**Thế bị lệch** (một bên thắng ≥ {int(metrics.SKEW_THRESHOLD * 100)}% số ván hợp lệ): "
            + ", ".join(f"`{r['opening_id']}` ({'Đỏ' if r['skew'] == 'red' else 'Đen'})" for r in skewed)
            + ". Vì mỗi thế đều được đánh cả hai màu nên không thiên vị agent nào, nhưng các thế này kém phân biệt sức mạnh.")
    else:
        add(f"Không có thế nào bị lệch (không bên nào thắng ≥ {int(metrics.SKEW_THRESHOLD * 100)}% số ván hợp lệ).")
    add("")

    # 8 ---------------------------------------------------------------
    add("## 8. Cảnh báo và giới hạn")
    add("")
    if warnings:
        lines.extend(f"- {w}" for w in warnings)
    else:
        add("Không có cảnh báo tự động nào.")
    add("")
    add("Giới hạn chung: Elo phụ thuộc prior 1 ván hòa ảo mỗi cặp; khoảng Wilson chỉ xấp xỉ khi có hòa; "
        "kết quả chỉ đúng với bộ thế khai cuộc, giới hạn nước đi và cấu hình agent của lần chạy này.")
    add("")

    # 9 ---------------------------------------------------------------
    add("## 9. Lệnh tái lập")
    add("")
    mode = manifest.get("mode", "full")
    mode_flag = {"smoke": " --smoke", "pilot": " --pilot"}.get(mode, "")
    config_path = manifest.get("config_path") or "configs/tournament.yaml"
    add(f"- Commit: `{git.get('commit') or 'không rõ'}` (nhánh `{git.get('branch') or 'không rõ'}`)"
        + (", **cây git bẩn lúc chạy**" if git.get("dirty") else ""))
    add(f"- Config: `{config_path}` (hash `{str(manifest.get('config_hash', ''))[:16]}`), master_seed "
        f"`{config.get('master_seed', '?')}`")
    add("")
    add("```powershell")
    add("cd project/backend")
    add(f"git checkout {git.get('commit') or '<commit>'}")
    add(f"python -m src tournament --config {config_path}{mode_flag} --workers {manifest.get('workers', 'N')}")
    add(f"python -m src tournament-report results/tournaments/{run_id}")
    add("```")
    add("")
    add("`tournament-report` chỉ đọc `manifest.json` và `games.jsonl` rồi sinh lại toàn bộ thư mục `summary/`; "
        "các file CSV sinh lại giống hệt từng byte.")
    add("")
    return "\n".join(lines)


def generate_report(run_dir: str | Path, *, plots: bool = True) -> dict[str, Any]:
    """Regenerate ``<run_dir>/summary`` and return what was written."""
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    records, notes = load_games(run_dir)
    config = manifest.get("config", {}) or {}
    collector = MetricsCollector(
        agents=list((manifest.get("agents") or {}).keys()),
        master_seed=int(config.get("master_seed", 0)),
        openings=list(config.get("openings", [])),
        anchor="random",
    )
    collector.add_matches(records)
    summary = collector.summarize()

    directory = run_dir / SUMMARY_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    written = collector.export(directory)

    plot_names: list[str] = []
    if plots:
        plot_names = make_plots(summary, directory / "plots")
    report = build_report(manifest, summary, run_dir, len(records), plot_names, notes)
    report_path = directory / "report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    written.append(report_path)
    written.extend(directory / "plots" / name for name in plot_names)
    return {
        "summary_dir": directory, "files": written, "completed": len(records),
        "total": manifest.get("total_games", len(records)), "plots": plot_names,
        "errors": summary["totals"]["error"],
    }
