# Lab05 — Toàn Thắng: kiểm tra và hoàn thiện Monte Carlo

> Nhánh này bổ sung phần kiểm tra chất lượng cho PR #53 (`Lab05/montecarlo`). Không triển khai lại phần Task 1–4 của các thành viên khác.

## Phạm vi đã đối chiếu với phân công

- Tic-Tac-Toe Bài 5: `tictactoe_pure_monte_carlo_search.ipynb` — Pure Monte Carlo với các mức `N = 10, 100, 1000`, bảng so sánh, nhận xét và thảo luận 4.
- Tic-Tac-Toe Bài 6: `tictactoe_monte_carlo_tree_search_restricted.ipynb` — thuật toán UCB1 có sẵn, thí nghiệm cùng mức `N`, so sánh với Pure Monte Carlo và thảo luận 5.
- Connect 4 Task 5: `connect4_montecarlo.py`, `test_connect4_montecarlo.py` và phần `## Task 5` trong `assignment_connect4.ipynb`.
- Tổng hợp báo cáo cuối: bản nháp `bao_cao.docx` đã được thêm ở PR #53; cần kiểm tra tính đầy đủ của báo cáo và xuất PDF trước khi nộp.

## Kiểm tra Monte Carlo

Hàm `playout` thực hiện một nước đi, sau đó mô phỏng ngẫu nhiên đến trạng thái kết thúc, trả về utility từ góc nhìn của người đang đi (`+1` thắng, `0` hòa, `-1` thua). Hàm `pmcs` chọn cột có utility trung bình cao nhất. `PMCSAgent` lưu thời gian thực hiện từng nước.

Thay đổi hoàn thiện của nhánh này:

1. Kiểm tra `N` phải là **số nguyên dương** để tránh ngân sách mô phỏng không hợp lệ.
2. Trong `playouts`, tái sử dụng `Position.from_board(board)` đã kiểm tra một lần cho cả batch. Mỗi lượt chơi vẫn tạo Position con mới qua `drop`, không sửa bàn cờ đầu vào.
3. Giữ nguyên thứ tự mô phỏng và chính sách chia đều từ PR #53, nhờ đó kết quả của các notebook đã lưu không bị thay đổi bởi refactor.
4. Bổ sung unit test cho ngân sách không hợp lệ, kết quả ngẫu nhiên khi batch/đơn lẻ dùng cùng seed, số playout thực sự và tính bất biến bàn cờ.
5. Bổ sung workflow GitHub Actions chạy test core, Minimax và Monte Carlo trên nhánh/PR.

### Phân biệt N yêu cầu với số lượt thật sự

**Quan trọng:** thuật toán hiện có chia đều `max(1, N // số_cột_hợp_lệ)` mô phỏng cho mỗi cột. Đó là quy ước *ngân sách xấp xỉ* của PR #53, **không** phải luôn đúng `N` mô phỏng. Ở trạng thái 6×7 trống (7 cột hợp lệ):

| N yêu cầu | Mô phỏng/cột | Tổng thực tế |
| ---: | ---: | ---: |
| 3 | 1 | 7 |
| 10 | 1 | 7 |
| 100 | 14 | 98 |
| 700 | 100 | 700 |
| 1000 | 142 | 994 |

Khi viết báo cáo, phải nói rõ đây là N *được yêu cầu*, còn ngân sách *thực tế* tùy số cột hợp lệ. Không được khẳng định đã chạy đúng 1.000 playout ở mọi thế cờ. Đổi sang phân bổ chính xác N sẽ làm thay đổi chuỗi ngẫu nhiên và buộc phải chạy lại toàn bộ bảng số liệu; vì vậy nhánh rà soát này **chưa** đổi chính sách đó.

### Kết quả đã có trong notebook (không phải số liệu chạy lại trên nhánh này)

Theo phần nhận xét lưu tại `assignment_connect4.ipynb` Task 5, thử 6 bàn × 20 lần mỗi mức N: số lần chọn đúng là **75/120 (N=10)**, **119/120 (N=100)** và **120/120 (N=1000)**. Cách đánh giá cho phép nước nằm trong tập nước mong đợi hoặc giữ được kết quả thắng (được định nghĩa trong notebook).

Theo `tictactoe_monte_carlo_tree_search_restricted.ipynb` Bài 6, trên 9 cặp trạng thái × N, UCB1 hơn Pure Monte Carlo ở 3 trường hợp, bằng 4 và kém 2. UCB1 phân bổ playout dựa trên điểm trung bình cộng phần thưởng khám phá `C sqrt(ln(tổng_lượt) / số_lượt_của_nước)`, thay vì chia đều; vì vậy không thể kết luận UCB1 luôn thắng Pure Monte Carlo ở mọi tình huống.

## Chạy kiểm thử

Từ thư mục `lab05 - game/lab05 - game`:

```bash
python -m pip install 'numpy>=1.23,<3'
python -m unittest -v test_connect4_core test_connect4_minimax test_connect4_montecarlo
```

Nếu muốn tái thực nghiệm notebook, cài đầy đủ `../../requirements.txt`, sau đó chạy notebook bằng Jupyter. Đầu ra đã lưu của notebook xuất phát từ PR #53, không đại diện cho một lần chạy mới trong nhánh này.

## Phần chờ để hoàn tất nộp bài

- [ ] Kiểm tra kết quả workflow unit test trên PR rà soát; sửa lỗi nếu CI phát hiện.
- [ ] Chờ `connect4_heuristic.py` của Tiến để chạy đối chứng Monte Carlo vs. Heuristic Alpha-Beta, hoặc ghi rõ giới hạn nếu không có trước hạn.
- [ ] Rà soát báo cáo `bao_cao.docx`, cập nhật số liệu/giải thích thuật toán, đảm bảo có tên đủ bốn thành viên.
- [ ] Chạy lại `assignment_connect4.ipynb` sau khi tích hợp Task 4, xuất bản HTML mới; xuất báo cáo PDF và đóng ZIP theo LMS.
- [ ] Merge PR rà soát vào `Lab05/montecarlo` **trước** khi merge PR #53 vào `main`, tránh chồng chéo cùng file.

*Lưu ý: đây là báo cáo kiểm tra mã nguồn và kết quả notebook đã lưu; không thay thế bước kiểm thử chạy thực tế hoặc xác nhận đã nộp LMS.*
