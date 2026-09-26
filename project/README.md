# Xiangqi AI

Nền tảng Python cho hệ thống AI Agent Cờ Tướng theo tài liệu trong `docs/`.
Python >= 3.10; phiên bản nền tảng không cần thư viện bên ngoài.

## Chạy nhanh

Từ thư mục `project`, trong PowerShell:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m xiangqi_ai play --seed 42 --max-plies 300 --output artifacts/games/random.json
python -m xiangqi_ai replay artifacts/games/random.json
python -m xiangqi_ai replay artifacts/games/random.json --all
python -m unittest discover -s tests -v
```

Có thể cài package bằng `python -m pip install -e .` rồi dùng lệnh `xiangqi` thay
cho `python -m xiangqi_ai` mà không cần `PYTHONPATH`.

## Phạm vi đã triển khai

- Luật di chuyển của 7 loại quân, cản mã, mắt tượng, sông, cung, ngòi pháo.
- Lọc nước đi để bảo vệ tướng; kiểm tra tướng đối mặt, chiếu bí và hết nước đi.
- State bất biến, environment reset/step/clone, action encoding và legal mask.
- RandomAgent dùng RNG riêng, match runner đo thời gian và lưu JSON replay.
- CLI render bàn cờ, xác minh replay bằng cách thực thi lại từng nước đi.

Chưa triển khai heuristic/search, mạng neural, training, giải đấu round-robin hoặc GUI.
Ưu tiên tiếp theo: scoring + Heuristic, benchmark đổi phe, Minimax/Alpha-Beta, MCTS,
rồi mới encoding tensor, neural network và training.

## Quy ước dữ liệu

- Board là tuple 90 ô, row-major: `square = row * 9 + col`; hàng 0 ở phía đen.
- Quân đỏ dương, quân đen âm, 0 là ô trống. Mã quân:
  `1=King, 2=Advisor, 3=Elephant, 4=Horse, 5=Rook, 6=Cannon, 7=Pawn`.
- Đỏ đi trước. Một ply là một lượt của một phe.
- `action_id = source * 90 + target`, encoding version 1, không gian 8100 phần tử.
  ID có source=target không hợp lệ; mask luôn loại bỏ chúng.
- `GameState` bất biến. `clone_state()` chia sẻ an toàn cùng state; áp dụng nước đi
  trả về state mới. History chứa board + lượt đi, bao gồm trạng thái hiện tại.
- `step()` trả `StepResult`; reward theo phe vừa đi: thắng +1, thua -1, hòa 0.
  Đây không phải Gymnasium adapter. Training phải xử lý riêng reward của đối thủ
  và transition từ lượt của agent đến lượt tiếp theo của chính agent.
- Agent nhận state và trả `AgentDecision`; match runner sở hữu ván thật.
  `legal_moves` sinh nước theo hình học/chiếu; environment và action mask còn kiểm
  tra kết thúc do hòa. Runner không gọi agent sau khi ván kết thúc.

## Bộ luật kết thúc `project_simplified_v1`

- Không có nước hợp lệ: phe đến lượt thua, kể cả không bị chiếu.
- Ba lần cùng board và lượt đi: hòa.
- 120 ply liên tiếp không ăn quân: hòa.
- Ưu tiên chiếu bí/hết nước đi trước điều kiện hòa.
- Không thực hiện adjudication trường chiếu/trường tróc theo luật giải đấu.
- Giới hạn `max_plies` của runner là **truncation**, không được tính thành hòa theo luật.
  Không có giới hạn thời gian cưỡng chế ở phiên bản này; thời gian quyết định chỉ được đo.

Giới hạn hòa cấu hình qua `RulesConfig`, được lưu trong từng replay. Kết quả JSON
lưu seed, tên agent, phiên bản encoding, cấu hình luật, action, thống kê thời gian,
trạng thái kết thúc và bàn cờ cuối. Cùng seed tái lập nước đi, không tái lập thời gian.

## Kiến trúc

`environment` chứa luật và state; `agents` chọn nước; `evaluation` điều phối trận;
`encoding` ánh xạ action; `storage` lưu/kiểm tra replay; `visualization` render text.
Engine không phụ thuộc agent, giao diện hoặc thư viện học sâu.

Engine hiện ưu tiên tính đúng: lọc bằng mô phỏng board và kiểm tra chiếu, lưu lịch sử
đầy đủ. Trước search sâu cần profiling chi phí legal-move generation và history;
chỉ tối ưu make/unmake hoặc hashing sau khi bộ test bảo vệ hành vi đã ổn định.
