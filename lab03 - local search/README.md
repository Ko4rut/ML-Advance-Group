# Lab 03 - Local Search

## Môi trường

Notebook được kiểm tra với Python 3.10.8. Dùng Python 3.10 trở lên; chọn cùng một phiên bản Python và cùng môi trường ảo khi chạy, đo thời gian và xuất kết quả. Các thư viện cần thiết được liệt kê trong [`requirements.txt`](requirements.txt).

Mở PowerShell tại thư mục gốc repository và chạy:

```powershell
cd "lab03 - local search"
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name lab03-local-search --display-name "Python (lab03-local-search)"
.\.venv\Scripts\python.exe -m jupyter lab
```

Trong mỗi notebook, chọn kernel **Python (lab03-local-search)**. Nếu PowerShell không tìm thấy `python`, cài Python 3.10+ và mở terminal mới. Không trộn package từ Python hệ thống, `.venv` ở repository gốc và `.venv` của lab này.

## Chạy notebook và dữ liệu

Khởi chạy Jupyter Lab từ thư mục `lab03 - local search` như lệnh trên, rồi mở notebook cần chạy. Chọn **Restart Kernel and Run All Cells** để chạy từ đầu; không cần tải dữ liệu hoặc module từ Internet. `L_maze.txt` và `maze_helper.py` nằm cạnh notebook mê cung.

Các notebook `n_queens.ipynb` và `traveling_salesman_problem.ipynb` là bài tập có các cell được để trống có chủ ý; phần cài đặt thuật toán và phân tích kết quả vẫn cần hoàn thành theo đề bài.

## Quy chuẩn tái lập và so sánh

- Dùng seed `1234` cho ví dụ tái lập. Với mỗi lần chạy benchmark, đặt lại seed trước khi tạo dữ liệu và chạy thuật toán; nếu dùng cả `random` và NumPy, đặt seed cho cả hai.
- So sánh trên cùng bộ dữ liệu/kích thước, cùng giới hạn lặp hoặc điều kiện dừng, cùng tham số và cùng dãy seed chạy độc lập. Không thay đổi tham số giữa các thuật toán trong cùng phép so sánh.
- Thực hiện 1 lần warm-up và 30 lần đo độc lập cho mỗi cấu hình. Với Numba/JIT, warm-up trước khi đo và không tính thời gian biên dịch vào thời gian chạy thuật toán.
- Đo thời gian bằng `time.perf_counter()`; chỉ bấm giờ phần thuật toán, không tính thời gian vẽ biểu đồ, in kết quả hay khởi tạo notebook.
- Ghi một dòng cho mỗi lần chạy với các trường: thuật toán, kích thước/bộ dữ liệu, seed, thành công, giá trị mục tiêu tốt nhất (ví dụ độ dài tour hoặc số xung đột), số bước/đánh giá trạng thái và thời gian (giây). Tổng hợp thời gian bằng median và IQR, đồng thời báo cáo tỷ lệ thành công và chất lượng nghiệm.
- Lưu kết quả dạng bảng CSV với tên cột thống nhất; ghi phiên bản Python, thư viện, tham số và seed trong phần báo cáo. Xóa output cũ, chạy lại toàn bộ notebook từ đầu trước khi xuất HTML hoặc nộp kết quả.

## Nội dung

### Ví dụ
- [Hill climbing trên mê cung](Maze_Example_LocalSearch.ipynb)
- [Tối ưu tính số xung đột N-Queens bằng Numba](n_queens_fast_conflict_calculation_with_numba.ipynb)
- [Lịch nhiệt độ simulated annealing](Simulated_Annealing_Schedules.ipynb)

### Bài tập
- [N-Queens](n_queens.ipynb)
- [Traveling Salesman Problem](traveling_salesman_problem.ipynb)
