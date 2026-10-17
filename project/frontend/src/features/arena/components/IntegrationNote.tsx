export function IntegrationNote({ connected }: { connected: boolean }) {
  let mark = "!";
  let title = "Backend offline";
  let description = "Chạy FastAPI tại localhost:8000 và làm mới trang để kết nối.";

  if (connected) {
    mark = "✓";
    title = "FastAPI connected";
    description = "Nước đi hợp lệ và trạng thái trận được lấy từ XiangqiEnv.";
  }

  return (
    <div className="roadmap-note">
      <span className="roadmap-mark">{mark}</span>
      <div>
        <strong>{title}</strong>
        <p>{description}</p>
      </div>
    </div>
  );
}

