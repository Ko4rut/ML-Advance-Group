export function IntegrationNote({ connected }: { connected: boolean }) {
  return <div className="roadmap-note">
    <span className="roadmap-mark">{connected ? "✓" : "!"}</span>
    <div><strong>{connected ? "FastAPI connected" : "Backend offline"}</strong>
      <p>{connected ? "Nước đi hợp lệ và trạng thái trận được lấy từ XiangqiEnv." : "Chạy FastAPI tại localhost:8000 và làm mới trang để kết nối."}</p>
    </div>
  </div>;
}
