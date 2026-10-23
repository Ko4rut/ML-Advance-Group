# Constraint Satisfaction Problem (CSP)

## Example
* [Solving the Graph Coloring using Simple Backtracking Search](CSP_graph_coloring_example.ipynb)

## Connection to Search
* The assignment [The n-Queens Problem](n_queens.ipynb) asks you to use local search to solve a CSP.

## N-Queens experiment logs

Mỗi batch tạo ID mới bằng `create_experiment_id()`. Cấu trúc:

```text
logs/<experiment_id>/
  experiment.json               # cấu hình Task 5 và starting boards
  runs.csv                      # một dòng metadata/metrics mỗi run
  summary.csv                   # tạo sau khi batch kết thúc
  histories/<algorithm_hash>/n8/
    run_000001.csv.gz           # chỉ dữ liệu từng bước, nén gzip
```

`save_log_result()` nối một dòng vào chỉ mục, không ghi lại toàn bộ bảng.
Hàm vẫn quét chỉ mục để kiểm tra số run; chỉ mục không chứa trajectory.
Gọi `summarize_experiment(experiment_id)` sau batch để tạo lại summary;
summary cũ bị loại bỏ khi thêm run để tránh đọc nhầm số liệu chưa cập nhật.
Nếu batch dừng giữa chừng, vẫn có thể tổng hợp các run đã lưu.
Chỉ dùng một tiến trình ghi vào một experiment tại một thời điểm.

```python
from helper import load_run_history
runs = pd.read_csv(f"logs/{experiment_id}/runs.csv")
run_id = runs.iloc[0]["run_id"]
history = load_run_history(experiment_id, run_id)
# Đọc từng phần nếu history lớn:
for chunk in load_run_history(experiment_id, run_id, chunksize=10_000):
    print(len(chunk))
```

Pandas tự giải nén `.csv.gz`. Metadata (board, seed, cấu hình) chỉ nằm trong
`runs.csv`; cột `history_csv` là đường dẫn tương đối trong experiment.
Dùng `save_history=False` nếu chỉ cần metrics, `plot_run=False` để bỏ ảnh.
Việc nén giảm dung lượng trên đĩa; thuật toán vẫn giữ history của run hiện tại
trong RAM. Tổng hợp cuối batch đọc bảng metrics vào RAM.

Log cũ không bị chuyển đổi hoặc xóa. Đọc CSV cũ như trước, nhưng tạo experiment
mới để ghi định dạng mới. Có thể nén/lưu trữ hoặc xóa riêng thư mục experiment
đã kiểm tra; ảnh tương ứng nằm ở `img/task_5_SA/<experiment_id>/`.

### Task 6 dùng chung định dạng log

`task6_analysis.py` đọc history Task 5 qua cột `history_csv` (gzip); vẫn đọc
được nguồn Task 5 cũ có history CSV chưa nén. Cache Task 6 mới tách khỏi bản v1:

```text
logs/
  <task5_experiment_id>/
    task6_all_algorithms_v2/
      runs.csv                  # chỉ mục HC + random restart, nối từng run
      histories/<algorithm_hash>/n<size>/run_000001.csv.gz
      analysis_runs.csv         # bảng phân tích chung SA + HC + random restart
      summary.csv               # tổng hợp tất cả thuật toán
      experiment.json
      comparison.csv, restart_comparison.csv, ...
  <task5_experiment_id>_task6_scalability_v2/
    runs.csv                    # SA cho n=12,16,20
    histories/<algorithm_hash>/n<size>/run_000001.csv.gz
    summary.csv
    experiment.json
img/task_6_all/<task5_experiment_id>/v2/
```

Chạy lại Task 6 sẽ dùng metrics đã lưu, chỉ đọc trajectory của run minh họa
(n=8, run=1) để vẽ. HC dùng lịch sử cải thiện, Efficient HC dùng lịch sử từng
lượt đề xuất, random restart dùng toàn bộ quá trình gồm các lần restart.
`metadata_json` trong chỉ mục giữ số evaluations, restart boundaries và các
thông tin phụ. Không tạo JSON chứa toàn bộ history cho từng run hay sao chép
history vào `examples.json` nữa. `analysis_runs.csv` là bảng metrics dẫn xuất;
không ghi đè chỉ mục `runs.csv` dùng để tiếp tục batch.

Cache v1 và nguồn Task 5 được giữ nguyên. Lần đầu dùng v2 sẽ chạy lại HC và SA
scalability; các lần tiếp theo dùng cache. Nếu đổi thuật toán hoặc tham số,
đổi tên/xóa thư mục cache v2 tương ứng trước khi chạy. Các số liệu thời gian
trong cache thuộc lần thực thi ban đầu.
