# Bàn giao: giải đấu vòng tròn, Phase 2 và 3

File này chưa commit, bạn tự xóa hoặc giữ. **Chưa commit gì, chưa push, chưa chạy giải thật 750 ván.**

## Đã làm gì

Phase 2 (thống kê và báo cáo):
- `python -m src tournament-report <run_dir> [--no-plots]` đọc `manifest.json` và `games.jsonl`, sinh lại toàn bộ `summary/`: `standings.csv`, `pair_results.csv`, `crosstable.csv`, `color_breakdown.csv`, `cost.csv`, `openings.csv`, `errors.csv` (chỉ khi có ván lỗi), `report.md` (tiếng Việt, 9 mục) và 3 biểu đồ PNG.
- Elo theo Bradley–Terry (thuật toán MM), neo `random` = 0, bootstrap 1000 lần, khoảng Wilson, tất cả tự viết bằng stdlib.
- Chạy lại `report` hai lần cho CSV và `report.md` giống hệt từng byte (đã kiểm bằng md5 trên smoke thật). Thiếu matplotlib thì bỏ qua biểu đồ, không crash. Run chưa xong thì báo "chưa hoàn tất x/N".

Phase 3 (test, tài liệu, hồi quy):
- Viết `TOURNAMENT.md` (hướng dẫn chạy trên Windows, schema `games.jsonl`, tái lập, giới hạn).
- Thêm test mới vào workflow CI `alphabeta-performance-tests.yml` (không có giải thật trong CI).

## File mới / sửa

Mới:
- `src/evaluation/report.py`
- `tests/evaluation/test_metrics.py` (27 test), `tests/evaluation/test_report.py` (14 test)
- `TOURNAMENT.md`, `TOURNAMENT_HANDOVER.md`

Sửa:
- `src/evaluation/metrics.py` (trước chỉ có khung `pass`; giữ `MetricsCollector` với `add_match`, `summarize`, `export`)
- `src/cli.py` (thêm lệnh `tournament-report`)
- `../../.github/workflows/alphabeta-performance-tests.yml` (thêm 5 module test)

## Kết quả test

- Toàn bộ `unittest discover`: **270 test, 269 pass, 1 lỗi cũ có sẵn** `tests.data_pipeline.test_silver_pipeline...test_sample_limit_preserves_full_bronze_provenance` (TypeError `read_jsonl`, lỗi cả trên `main`, không liên quan).
- Danh sách test trong workflow CI (kèm 5 module mới): 152 test, tất cả pass.
- `python -m src.evaluation.run_mcts_matches --games 1 --opponents random --max-plies 20`: chạy bình thường (ghi file ra thư mục tạm).
- `python -m src batch --red rules --black random --games 2 --max-plies 10`: chạy bình thường.
- `compare_alphabeta` (bước microbenchmark của CI): chạy bình thường.
- Smoke thật (`--smoke --workers 5`, 60 ván, 2 phút 17 giây): `results/tournaments/smoke_20261111-211132_e1a86c2c/`. Đã đọc lại `report.md`, số liệu khớp CSV, định dạng ổn.

## Lệnh chạy giải thật đề xuất

```powershell
cd project/backend
powercfg /change standby-timeout-ac 0
python -m src tournament --config configs/tournament.yaml --workers 5
# nếu bị ngắt: python -m src tournament --resume results\tournaments\<run_id>
python -m src tournament-report results\tournaments\<run_id>
```

Ước lượng thời gian (theo pilot bạn đã có): khoảng 1 giờ với 5 worker, khoảng 4 giờ 55 phút nếu chạy tuần tự. Nên commit code trước khi chạy để report không bị cảnh báo "cây git bẩn".

## Giới hạn còn lại (đã nằm trong report và `TOURNAMENT.md`)

- Khoảng Wilson chỉ xấp xỉ khi có hòa; 15 cặp không hiệu chỉnh so sánh nhiều lần.
- Elo phụ thuộc prior 1 ván hòa ảo mỗi cặp.
- Ván truncated tính 0.5 điểm; thứ hạng có thể đổi nếu loại chúng (report tự báo).
- `heuristic` và `mcts` dùng đồng hồ nên chỉ tái lập về mặt thống kê, và số worker ảnh hưởng đến chúng. Muốn mcts tái lập chính xác thì đặt `max_simulations`.
- `tabular_q`: thực đo ~96.5% nước đi rơi về `rules`, nên gần như là `rules`. Trong smoke, `rules vs tabular_q` có 2 ván đổi màu trùng hệt nhau (`distinct` 3/4); report tự cảnh báo.
- Chưa xác minh được web đang dùng checkpoint local hay file trên Drive.
- Phát hiện thêm: `heuristic` gần như luôn dừng vì hết 1 giây (trung vị 1.000 giây) với trung bình chỉ ~14 node, nên kết quả của nó rất nhạy với tốc độ máy. Không sửa vì ngoài phạm vi (không đổi agent), nhưng bạn nên biết khi giải thích kết quả.
- Smoke giới hạn 40 nước nên 78% ván là truncated, đừng dùng kết quả smoke để kết luận về sức mạnh.

## Quyết định tôi tự chọn (vì bạn đang ngủ)

- Khoảng Wilson và `score_pct` trong CSV tính theo phần trăm (0–100), còn hàm `wilson_interval` trả về tỉ lệ (0–1).
- `A_as_red_score` / `A_as_black_score` là tổng điểm của A khi cầm Đỏ / Đen (không phải phần trăm).
- Cột `games` trong `standings.csv` tính cả ván lỗi; điểm và phần trăm chỉ tính ván hợp lệ.
- `cost.csv` tính mọi nước do agent đi, kể cả trong ván lỗi; `distinct_games` không tính ván lỗi.
- Xếp hạng đồng điểm dùng hạng chung (1, 2, 2, 4).
- Dòng `TOTAL` trong `color_breakdown.csv` là lợi thế đi trước: điểm của ghế Đỏ và ghế Đen trên toàn giải.
- Cột thêm `skew` trong `openings.csv` (thế mà một bên thắng ≥ 70% ván hợp lệ).
- `report.md` chỉ chèn ảnh khi biểu đồ vừa được sinh ở lần chạy đó, nên bản `--no-plots` khác bản có biểu đồ.
- `tournament-report` không sửa `games.jsonl` (dòng cuối bị cắt dở chỉ bị bỏ qua và ghi chú vào report).
- `tabular_q` fallback bị coi là "cao" (cảnh báo mạnh) khi ≥ 50%.
- Chỉ thêm test mới vào CI, không thêm giải đấu thật.

## Gợi ý commit message (bạn tự commit)

```
feat(eval): add tournament metrics, Elo and Vietnamese report generator
```
```
docs(eval): add TOURNAMENT.md and run new tournament tests in CI
```
