### Nhận xét khả năng mở rộng

**Simulated Annealing:** mô hình thực nghiệm T(n) ≈ C·n^2.42 (R²=0.930); có thể mô tả mức tăng quan sát được là O(n^2.42) trong phạm vi đo, không coi đây là cận độ phức tạp lý thuyết.

**Steepest-ascent HC:** mô hình thực nghiệm T(n) ≈ C·n^3.82 (R²=1.000); có thể mô tả mức tăng quan sát được là O(n^3.82) trong phạm vi đo, không coi đây là cận độ phức tạp lý thuyết.

Tại n=20, **Simulated Annealing** dừng nhanh hơn trong phép đo này (15.821 ms trung bình; tỷ lệ thành công 3%). Cần đối chiếu tỷ lệ thành công của cả hai trong bảng trước khi chọn thuật toán giải bài toán.

SA đánh giá một hàng xóm mỗi lượt; hàm conflicts hiện tại tốn O(n), nên với giới hạn K lượt, chi phí tìm kiếm là O(Kn). K cố định và nhiều run lớn dùng hết K có thể khiến đường thời gian gần tuyến tính, dù khả năng tìm lời giải suy giảm. Steepest HC duyệt n(n−1) hàng xóm mỗi vòng, mỗi lần tốn O(n), nên tốn O(n³) mỗi vòng; tổng chi phí còn phụ thuộc số bước cải thiện. Các run nhỏ dừng sớm và chi phí cố định làm số mũ đo được khác mô hình lý thuyết.

Không khẳng định một thuật toán luôn tốt nhất cho n lớn từ năm điểm đo. Kết quả n=4,8 của SA được tái sử dụng từ Task 5, nên chênh lệch tải máy giữa các lần thực nghiệm cũng có thể ảnh hưởng thời gian. Muốn đánh giá thời gian tìm lời giải cần tăng ngân sách hoặc dùng tiêu chí chất lượng chung và thực nghiệm riêng.