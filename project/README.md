# Xiangqi AI

Nền tảng Python cho hệ thống AI Agent Cờ Tướng theo tài liệu trong `docs/`, kèm frontend Next.js cho Xiangqi AI Arena.

## Cấu trúc project

```text
project/
├── assets/       # tài nguyên dùng chung, đặc biệt cho frontend
├── configs/
├── docs/
├── artifacts/
├── backend/      # Python environment, agents, training, evaluation và CLI
└── frontend/     # Next.js App Router
```

Python backend yêu cầu Python >= 3.10 và hiện không cần thư viện ngoài cho phần core.

## Backend

Từ thư mục `project/backend`:

```powershell
python -m unittest discover -s tests -v
python -m src show
python -m src random --seed 42 --max-plies 20
python -m src random --seed 42 --max-plies 20 --delay 0.3
```

Code Python nằm trực tiếp trong package `src`, không có package trung gian `xiangqi_ai`.
Ví dụ import environment: `from src.environment import XiangqiEnv`.

Có thể cài backend ở chế độ editable:

```powershell
cd project/backend
python -m pip install -e .
xiangqi show
xiangqi random --max-plies 20
```

## Frontend

Frontend dùng Next.js App Router, TypeScript và Tailwind CSS. Từ `project/frontend`:

```powershell
npm install
npm run dev
```

Frontend hiện mới là scaffold để phát triển Xiangqi AI Arena. Tài nguyên bàn cờ/quân cờ tiếp tục được giữ ở `project/assets` thay vì chuyển vào backend.

## Phạm vi đã triển khai

- Luật di chuyển của 7 loại quân, cản mã, mắt tượng, sông, cung, ngòi pháo.
- Lọc nước đi để bảo vệ tướng; kiểm tra tướng đối mặt, chiếu bí và hết nước đi.
- State bất biến và environment hỗ trợ reset/step/clone.
- Giao diện agent chung, RandomAgent có seed tái lập được.
- RuleBasedAgent baseline đã được triển khai và kiểm thử.
- CLI xem state và chạy một ván random trên terminal.
- Next.js frontend đã được khởi tạo để thay thế Streamlit.

Các package encoding, search, mạng neural, training, evaluation và storage hiện mới
là khung cho các giai đoạn tiếp theo.

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

## Kiến trúc backend

`environment` chứa luật và state; `agents` chọn nước; `evaluation` điều phối trận;
`encoding` ánh xạ action; `storage` lưu/kiểm tra replay; `visualization` giữ các
tiện ích render/debug không phụ thuộc web UI.

Engine không phụ thuộc agent, frontend hoặc thư viện học sâu.

Engine hiện ưu tiên tính đúng: lọc bằng mô phỏng board và kiểm tra chiếu, lưu lịch sử
đầy đủ. Trước search sâu cần profiling chi phí legal-move generation và history;
chỉ tối ưu make/unmake hoặc hashing sau khi bộ test bảo vệ hành vi đã ổn định.
