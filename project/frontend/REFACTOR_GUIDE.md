# Frontend Refactor Guide

## Quy tắc áp dụng

- Không dùng arrow function (`=>`) trong TypeScript/TSX.
- Không dùng toán tử ba ngôi (`condition ? trueValue : falseValue`). Dùng `if/else`.
- Callback phải là hàm có tên: `function renderSquare(...)`, `function handleSelect()`.
- JSX props dài tách dòng, mỗi prop một dòng. Components nhiều khối tách thành component con.
- Không gọi `fetch` trực tiếp từ component/hook: chỉ gọi các API module.
- Không sửa `main` trực tiếp; mọi thay đổi thực hiện qua feature/refactor branch.

## API modules

`src/features/arena/api/client.ts` duy nhất quản lý `fetch`, `API_BASE_URL`, HTTP error và network error.

`api/agents.ts`: lấy catalog và agent availability.

`api/matches.ts`: tạo trận, lấy snapshot, reset trận.

`api/moves.ts`: gửi nước đi của người và step AI. Cả hai gửi `expected_ply` để backend phát hiện request stale.

`api/types.ts`: kiểu dữ liệu chung tương ứng schema FastAPI.

`lib/api.ts` chỉ là lớp tương thích (deprecated); code mới import module API theo chức năng.

## Hook

**useAgentCatalog** tải danh sách agent khi mount, chỉ đánh dấu agent có `available = true`. Cleanup tránh set state sau khi unmount.

**useArenaMatch** quản lý snapshot, chọn phe, chọn ô, reset, modal và autoplay. Các bước quan trọng:

1. `chooseAgent`: đổi người chơi, invalidates trận trước, yêu cầu tạo trận mới.
2. `createMatch`: kiểm tra agent khả dụng, POST trận, áp dụng snapshot từ server.
3. `selectSquare`: chỉ cho chọn quân phe tới lượt có legal move; lần nhấn thứ hai gửi nước đi.
4. `moveHuman`: chuyển chỉ số ô từ UI sang engine, gửi `expected_ply`, nhận snapshot mới.
5. `startBattle` và `agentAutoplayEffect`: tự gọi endpoint step từng lượt sau 550ms, dừng khi tới lượt người hoặc kết thúc.
6. `handleApiFailure`: nếu HTTP 409, tải lại snapshot để tránh UI đang giữ `ply` cũ.

`busyRef` ngăn gọi API song song. `generationRef` giúp bỏ response muộn của trận cũ. Bàn cờ lấy từ FastAPI; frontend không tự tính luật cờ.

## Kiểm tra sau khi checkout

```powershell
cd project/frontend
npm ci
npm run lint
npx tsc --noEmit
npm run build
```

Kiểm thử thủ công: tạo trận Human vs Human, xem dấu nước đi hợp lệ, đi quân, reset; sau đó kiểm tra Human vs Random, Random vs Rule-Based và trạng thái mất kết nối API.

Quy tắc cấm arrow/ternary được kiểm tra bằng ESLint (`no-restricted-syntax`).
