# Lab 03 — Tìm kiếm cục bộ

Lab này tìm hiểu cách dùng Hill Climbing và Simulated Annealing để giải N-Queens, bài toán người bán hàng (TSP) và minh họa tìm kiếm trên mê cung.

## Nhóm đã làm được gì?

- **N-Queens:** cài đặt ba biến thể Hill Climbing, random restart và Simulated Annealing; thử các lịch giảm nhiệt, so sánh kết quả, vẽ quá trình hội tụ và đo thời gian theo kích thước bàn cờ. Task 6 dùng 100 lần chạy cho mỗi thuật toán/kích thước được báo cáo, có lưu log và hình để xem lại.
- **TSP:** đã có các thuật toán Hill Climbing, random restart, Simulated Annealing, bảng so sánh và phần Genetic Algorithm.
- **Ví dụ bổ trợ:** minh họa cực trị cục bộ trên mê cung, các lịch giảm nhiệt và cách tăng tốc tính xung đột bằng Numba.

Phần nâng cao về toán tử di chuyển và Genetic Algorithm trong **notebook N-Queens** vẫn chưa hoàn thành.

## Các thành phần trong thư mục

| File / thư mục | Nội dung |
| --- | --- |
| [n_queens.ipynb](n_queens.ipynb) | Bài N-Queens chính; Task 1–4 là HC, Task 5 là SA, Task 6 là phân tích kết quả. |
| [traveling_salesman_problem.ipynb](traveling_salesman_problem.ipynb) | Các thuật toán tìm đường đi ngắn cho TSP. |
| [Maze_Example_LocalSearch.ipynb](Maze_Example_LocalSearch.ipynb) | Ví dụ tìm kiếm cục bộ trên mê cung. |
| [Simulated_Annealing_Schedules.ipynb](Simulated_Annealing_Schedules.ipynb) | Minh họa và so sánh các lịch giảm nhiệt. |
| [n_queens_fast_conflict_calculation_with_numba.ipynb](n_queens_fast_conflict_calculation_with_numba.ipynb) | Ví dụ tăng tốc tính số xung đột bằng Numba. |
| [simulated_annealing.py](simulated_annealing.py), [annealing_schedules.py](annealing_schedules.py) | Thuật toán SA cho N-Queens và các hàm nhiệt độ. |
| [helper.py](helper.py) | Tạo/vẽ bàn cờ, tính xung đột, lưu log và tổng hợp kết quả. |
| [task6_analysis.py](task6_analysis.py) | Chạy hoặc đọc lại thực nghiệm Task 6, xuất bảng và biểu đồ. |
| [maze_helper.py](maze_helper.py), [L_maze.txt](L_maze.txt) | Hàm hỗ trợ và dữ liệu mê cung. |
| [logs/](logs/), [img/](img/) | Kết quả từng lần chạy, lịch sử tìm kiếm, bảng tổng hợp và hình thực nghiệm. |
| [docs/](docs/), [wp1_figures/](wp1_figures/) | Tài liệu theo thành viên và hình minh họa mê cung. |
| Các file PDF, DOCX ở thư mục này | Tài liệu bài lab và nội dung báo cáo. |

## Cách xem và chạy

**Chỉ xem kết quả:** mở `n_queens.ipynb` bằng VS Code hoặc Jupyter, rồi xem các bảng và hình đã lưu ở Task 5–6. Không cần chạy lại thuật toán.

**Muốn chạy lại:** dùng môi trường Python của project, mở terminal tại thư mục gốc repository và chạy:

```bash
python -m pip install -r requirements.txt
cd "lab03 - local search"
python -m jupyter lab
```

Mở notebook cần xem, chọn kernel đúng môi trường đã cài thư viện và chạy các cell từ trên xuống. File [requirements.txt](../requirements.txt) nằm ở **thư mục gốc repository**.

Riêng **Task 6 của N-Queens**, chạy các cell import và định nghĩa thuật toán ở Task 1–4 trước; không cần chạy lại Task 5. Task 6 tự chọn bộ log Task 5 mới nhất đã hoàn tất theo timestamp trong tên thư mục, bỏ qua lần chạy dở và thư mục scalability. Không cần sửa tên log sau mỗi lần chạy Task 5. Nếu chưa có log, chạy Task 5 trước. Muốn tái lập một lần chạy cụ thể, dùng `resolve_task5_source(Path("logs/<experiment_id>"))`. Task 6 tự tạo các run SA scalability còn thiếu và dùng lại cache ở những lần sau. Giữ các thư mục log này khi chia sẻ bài. Kết quả HC đã có sẽ được đọc từ cache; nếu sửa thuật toán, đổi hoặc xóa thư mục cache `task6_all_algorithms_v1` để đo lại.

Chạy lại Task 5 sẽ tạo một bộ log mới, nhưng Task 6 vẫn dùng bộ log đã chỉ định. Đường dẫn log và ảnh được in trong notebook để dễ tìm.
