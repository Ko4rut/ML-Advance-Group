# Gợi ý nước đi hợp lệ — UI & API Contract

**Dự án:** Xiangqi AI Arena · **Trạng thái:** tài liệu thiết kế (chưa triển khai API).

## 1. Quyết định kiến trúc

**Chỉ có backend quyết định nước nào hợp lệ.** Backend Python đã có:
- `XiangqiEnv.get_legal_actions() -> list[Move]`: liệt kê nước hợp lệ cho **phe đang đến lượt**, đã lọc những nước khiến tướng phe mình bị chiếu.
- `XiangqiEnv.step(Move(source, target))`: kiểm tra và áp dụng nước, trả về `StepResult` (state/reward/terminated/result).
- `Move` sử dụng chỉ số ô `0..89`, `index = row * 9 + col`.

Frontend Next.js chỉ **render các điểm đến do backend trả về**, không chép lại luật di chuyển Mã/Tượng/Pháo/Tướng và không tin một nước đi chỉ vì UI đã highlight nó. Các AI agent cũng dùng cùng luật hợp lệ trên backend; mục đích của dấu gợi ý là "nước hợp lệ", **không phải "nước tốt nhất theo AI"**.

## 2. Trải nghiệm người dùng

1. Bắt đầu ván: server trả bàn cờ, `side_to_move`, `ply`, `legal_moves`.
2. Nếu tới lượt người chơi, click quân **thuộc đúng phe**; UI lọc `legal_moves` theo `source`.
3. Ô trống hợp lệ: chấm tròn nhỏ (đường kính khoảng 9–11 px, hơi trong suốt).
4. Ô có quân địch ăn được: viền tròn mảnh bao quanh quân đích (không phủ màu lên mặt quân).
5. Quân đang chọn: viền mảnh, nhất quán với hover hiện tại; có thể click lại để bỏ chọn.
6. Click ô đích hợp lệ: gọi `POST /matches/{id}/moves`, rồi dùng **state mới từ server** để render.
7. Click ô không hợp lệ: giữ nguyên trạng thái, không gửi lệnh di chuyển. Khi AI đang suy nghĩ/đến lượt hoặc ván kết thúc, không cho người chơi thao tác.
8. Nếu API trả lỗi nước đi cũ (`409`) hoặc không hợp lệ (`422`), lấy lại snapshot, xóa selection/markers và báo ngắn.

Ký hiệu gợi ý nên là **CSS overlay** ở giao điểm bàn cờ, độc lập với ảnh quân và tỉ lệ scale quân cờ. Không tăng `width` quân cờ khi hover; không dùng lớp nền trắng lớn.

## 3. Điều đặc biệt quan trọng: hướng bàn cờ

Hiện UI thử nghiệm gán **đỏ ở trên** (`top`) và **đen ở dưới** (`bottom`). Nhưng `GameState.initial()` của backend có **đen ở row=0** và **đỏ ở row=9**. Backend quy ước **đỏ đi trước**.

Để giữ thiết kế giao diện hiện tại mà vẫn dùng luật backend, phải chuyển đổi **hàng** ở lớp hiển thị (không chỉnh engine):

```ts
const toEngineSquare = (displayIndex: number) => {
  const row = Math.floor(displayIndex / 9);
  const col = displayIndex % 9;
  return (9 - row) * 9 + col;
};
const toDisplaySquare = toEngineSquare; // Phép biến đổi này tự nghịch đảo.

// viewBoard[displayRow * 9 + col] = serverBoard[(9 - displayRow) * 9 + col]
```

Ví dụ: từ pawn **đỏ** `source=54` đến `target=45` (chỉ số engine) => phía giao diện là `27 → 36`. Khi click trả về backend **54 → 45**, không gửi chỉ số hiển thị. Nếu sau này chuyển thiết kế sang đỏ ở dưới thì bỏ phép đảo row và cập nhật nhãn màu.

## 4. API MVP đề xuất

Các đường dẫn dưới đây là **đề xuất**, không phải endpoint đã tồn tại. Có thể đặt prefix `/api` hoặc `/api/v1` khi xây FastAPI.

| Method | Path | Chức năng |
|---|---|---|
| GET | `/agents` | Danh sách agent + khả dụng (`available`), tránh cho chọn skeleton như agent đã chạy được |
| POST | `/matches` | Tạo ván với 2 đối thủ; trả snapshot đầu |
| GET | `/matches/{id}` | Snapshot đầy đủ để refresh và đồng bộ |
| POST | `/matches/{id}/moves` | Người chơi yêu cầu đi một nước; backend kiểm tra rồi trả snapshot mới |
| POST | `/matches/{id}/step` | Tùy chọn: yêu cầu AI đi đúng **một** nước (AI vs AI hoặc Human vs AI) |

**Khuyến nghị:** nhúng `legal_moves` vào **snapshot** của `POST /matches`, `GET /matches/{id}` và `POST /moves` (danh sách thường nhỏ). Như vậy chọn quân trên giao diện hoàn toàn tức thì, **không phải gọi API sau mỗi click**. Endpoint `GET /matches/{id}/legal-moves?source=...` chỉ là lựa chọn thay thế nếu muốn tải riêng.

### 4.1. Dữ liệu ví dụ

```http
POST /matches
Content-Type: application/json

{
  "red": {"type": "human"},
  "black": {"type": "agent", "agent_id": "random"}
}
```

```json
{
  "match_id": "match_123",
  "state": {
    "board": [/* mảng 90 số nguyên: 0, +1..+7 đỏ, -1..-7 đen */],
    "side_to_move": "red",
    "ply": 0,
    "status": "playing",
    "winner": null,
    "legal_moves": [
      {"source": 54, "target": 45}
    ]
  }
}
```

Mảng `legal_moves` chỉ minh họa **một phần**; API thực tế phải trả **toàn bộ** nước hợp lệ đang có. `board` ví dụ cần đúng 90 phần tử.

```http
POST /matches/match_123/moves
Content-Type: application/json

{"source": 54, "target": 45, "expected_ply": 0}
```

Server kiểm tra `expected_ply` chống click trùng / nhận trạng thái cũ, kiểm tra phe tới lượt và move nằm trong `env.get_legal_actions()`, rồi `env.step(Move(54,45))`. Kết quả trả snapshot đầy đủ gồm `ply=1`, `side_to_move="black"`, `legal_moves` mới. Đầu ra `status` có thể là `playing | finished` và `result.reason` khi kết thúc.

**Quy ước mã quân:** `0` trống; dấu **dương = đỏ**, âm = đen; `1 King, 2 Advisor, 3 Elephant, 4 Horse, 5 Rook, 6 Cannon, 7 Pawn`.

**Lỗi:** `404` không tìm thấy trận; `409` `expected_ply` lệch hoặc không đúng lượt; `422` nước đi không hợp lệ; `503` agent chưa được triển khai. Lỗi không làm thay đổi state.

## 5. Vị trí code trong project

```text
project/
  backend/
    src/environment/               # ĐÃ CÓ: XiangqiEnv, Move, GameState, legal moves
    src/api/                       # ĐỀ XUẤT TẠO: FastAPI routes, schemas, match service
  frontend/
    src/features/arena/
      components/ChessBoard.tsx    # Hiển thị quân + overlays điểm đến
      hooks/useArenaPreview.ts     # Sẽ thay bằng useMatch / dữ liệu API
      lib/board.ts                 # Mapping index engine ↔ display
      lib/api.ts                   # Client HTTP được thêm sau
      types.ts                     # MatchState, LegalMove, v.v.
```

Các lớp cần tách riêng: `MatchService` sở hữu `XiangqiEnv` theo `match_id`; `API` xác thực và trả JSON; `Frontend` không lưu game state làm nguồn chân lý. MVP có thể lưu các trận trong RAM; sau này cần persistence/cleanup cho các trận dài.

## 6. Danh sách kiểm thử chấp nhận

- Mở trận mới: `legal_moves` thuộc đỏ, không thấy nước của đen.
- Chọn Tốt đỏ ban đầu: đánh dấu đúng ô tiến một bước; sau khi đi, cập nhật lượt.
- Chọn quân sai phe, click ô không hợp lệ: không gửi `POST /moves`.
- Di chuyển ăn quân: hiển thị viền mảnh quanh mục tiêu có quân địch, không đè icon.
- Kiểm tra cản Mã, mắt Tượng, ngòi Pháo, giới hạn cung/sông, hai Tướng đối mặt, tự chiếu.
- Khi hết trận, `legal_moves=[]`, không nhận nước đi mới.
- Hai request cùng `expected_ply`: chỉ một request được áp dụng.
- Test hướng bàn cờ: engine index 54 hiển thị index 27, click target display 36 gửi engine index 45.
- Agent chưa triển khai: `GET /agents` có `available=false`, frontend không giả vờ cho chạy.

## 7. Thứ tự triển khai thực tế

1. Viết và test `MatchService` bọc `XiangqiEnv`; xác định màu/đối thủ theo engine.
2. Tạo `POST /matches`, `GET /matches/{id}`, `POST /moves`; dùng bộ test hiện có của engine.
3. Nối `useMatch` và `ChessBoard`: click nguồn, render dots/rings, click đích, cập nhật snapshot.
4. Kết nối `RandomAgent` bằng `POST /step`; sau đó mới đến các AI khác.
5. Bổ sung autoplay/WebSocket, thống kê và lưu trận.

**Chốt:** Gợi ý nước đi hợp lệ là dữ liệu từ luật cờ trên backend + overlay nhỏ trên frontend. Không cần viết logic luật cờ lần hai ở Next.js.
