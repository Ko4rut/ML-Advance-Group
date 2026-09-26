# Xiangqi AI

Nền tảng Python cho hệ thống AI Agent Cờ Tướng theo tài liệu trong `docs/`.
Python >= 3.10; phiên bản nền tảng không cần thư viện bên ngoài.

## Chạy nhanh

Từ thư mục `project`, trong PowerShell:

```powershell
python -m unittest discover -s tests -v
```

Code nằm trực tiếp trong package `src`, không có package trung gian `xiangqi_ai`.
Ví dụ import environment: `from src.environment import XiangqiEnv`.

Có thể cài project ở chế độ editable bằng `python -m pip install -e .`.

## Phạm vi đã triển khai

- Luật di chuyển của 7 loại quân, cản mã, mắt tượng, sông, cung, ngòi pháo.
- Lọc nước đi để bảo vệ tướng; kiểm tra tướng đối mặt, chiếu bí và hết nước đi.
- State bất biến và environment hỗ trợ reset/step/clone.

Các package encoding, agent, search, mạng neural, training, evaluation, storage,
visualization và CLI hiện mới là khung cho các giai đoạn tiếp theo.

## Quy ước dữ liệu

- Board là tuple 90 ô, row-major: `square = row * 9 + col`; hàng 0 ở phía đen.
- Quân đỏ dương, quân đen âm, 0 là ô trống. Mã quân:
  `1=King, 2=Advisor, 3=Elephant, 4=Horse, 5=Rook, 6=Cannon, 7=Pawn`.
- Đỏ đi trước. Một ply là một lượt của một phe.
- `GameState` bất biến. `clone_state()` chia sẻ an toàn cùng state; áp dụng nước đi
  trả về state mới. History chứa board + lượt đi, bao gồm trạng thái hiện tại.
- `step()` trả `StepResult`; reward theo phe vừa đi: thắng +1, thua -1, hòa 0.
  Đây không phải Gymnasium adapter. Training phải xử lý riêng reward của đối thủ
  và transition từ lượt của agent đến lượt tiếp theo của chính agent.
- `legal_moves` sinh nước theo hình học rồi loại các nước khiến tướng phe đang đi
  bị chiếu. Environment không trả nước đi sau khi ván đã kết thúc.

## Bộ luật kết thúc `project_simplified_v1`

- Không có nước hợp lệ: phe đến lượt thua, kể cả không bị chiếu.
- Ba lần cùng board và lượt đi: hòa.
- 120 ply liên tiếp không ăn quân: hòa.
- Ưu tiên chiếu bí/hết nước đi trước điều kiện hòa.
- Không thực hiện adjudication trường chiếu/trường tróc theo luật giải đấu.
Giới hạn hòa được cấu hình qua `RulesConfig`.

## Kiến trúc

`environment` chứa luật và state; `agents` chọn nước; `evaluation` điều phối trận;
`encoding` ánh xạ action; `storage` lưu/kiểm tra replay; `visualization` render text.
Engine không phụ thuộc agent, giao diện hoặc thư viện học sâu.

Engine hiện ưu tiên tính đúng: lọc bằng mô phỏng board và kiểm tra chiếu, lưu lịch sử
đầy đủ. Trước search sâu cần profiling chi phí legal-move generation và history;
chỉ tối ưu make/unmake hoặc hashing sau khi bộ test bảo vệ hành vi đã ổn định.
