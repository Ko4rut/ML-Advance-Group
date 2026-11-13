import { test, expect } from "@playwright/test";

const top = 'section[aria-label="Chọn đối thủ phía trên"]';
const bottom = 'section[aria-label="Chọn đối thủ phía dưới"]';

// Catalog fixtures exercise UI behavior without requiring weights in ordinary CI.
test("unavailable checkpoint is disabled with an understandable reason", async ({ page }) => {
  await page.route("**/api/v1/agents", (route) => route.fulfill({ json: [
    { id: "random", name: "Random", available: true },
    { id: "neural", name: "Neural MCTS", available: false,
      unavailable_reason: "Neural MCTS unavailable: checkpoint not found; provision weights and restart API" },
  ] }));
  await page.goto("/");
  await expect(page.locator(bottom).getByRole("button", { name: /Neural MCTS/ })).toBeDisabled();
  await expect(page.locator(bottom).getByRole("note")).toContainText("checkpoint not found");
  await expect(page.locator(bottom).getByRole("button", { name: "Random" })).toBeEnabled();
});

test("match creation errors remain visible", async ({ page }) => {
  await page.route("**/api/v1/agents", (route) => route.fulfill({ json: [
    { id: "neural", name: "Neural MCTS", available: true },
  ] }));
  await page.route("**/api/v1/matches", (route) => route.fulfill({ status: 503, json: {
    detail: "Neural MCTS unavailable: invalid checkpoint SHA-256",
  } }));
  await page.goto("/");
  await page.locator(bottom).getByRole("button", { name: "Neural MCTS" }).click();
  await page.locator(".controls-panel").getByRole("button", { name: "Tạo trận đấu" }).click();
  await expect(page.locator(".board-caption")).toContainText("invalid checkpoint SHA-256");
});

test("catalog connection error is shown", async ({ page }) => {
  await page.route("**/api/v1/agents", (route) => route.abort());
  await page.goto("/");
  await expect(page.locator(".board-caption")).toContainText("Không kết nối được FastAPI");
});

// These are explicitly real HTTP/browser tests; run with the provisioned backend.
for (const neuralSide of ["red", "black"] as const) {
  test(`real pretrained Neural MCTS plays ${neuralSide} through the browser`, async ({ page }) => {
    test.skip(process.env.XIANGQI_NEURAL_REQUIRE_REAL !== "1", "requires a running backend with a real checkpoint");
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    await page.locator(neuralSide === "red" ? top : bottom).getByRole("button", { name: "Neural MCTS" }).click();
    const created = page.waitForResponse((r) => r.url().endsWith("/api/v1/matches") && r.request().method() === "POST");
    await page.locator(".controls-panel").getByRole("button", { name: "Tạo trận đấu" }).click();
    let state = await (await created).json();
    expect(state.opening_mode).toBe("none");
    if (neuralSide === "red") {
      const step = page.waitForResponse((r) => r.url().endsWith("/step"));
      await page.getByRole("button", { name: /Chạy AI/ }).click();
      const reply = await step;
      expect(reply.status()).toBe(200);
      const next = await reply.json();
      expect(state.legal_moves).toContainEqual(next.last_move);
      state = next;
      await expect(page.getByRole("button", { name: /Chạy AI/ })).toBeDisabled();
    }
    // Pick a legal HUMAN action from the authoritative snapshot, then click board squares.
    const move = state.legal_moves[0];
    const display = (index: number) => (9 - Math.floor(index / 9)) * 9 + index % 9;
    const humanMove = page.waitForResponse((r) => r.url().endsWith("/moves"));
    const aiMove = page.waitForResponse((r) => r.url().endsWith("/step"));
    await page.locator(".board-intersection").nth(display(move.source)).click();
    await page.locator(".board-intersection").nth(display(move.target)).click();
    const afterHuman = await (await humanMove).json();
    expect(afterHuman.last_move).toEqual(move);
    const reply = await aiMove;
    expect(reply.status()).toBe(200);
    const afterAI = await reply.json();
    expect(afterHuman.legal_moves).toContainEqual(afterAI.last_move);
    await expect(page.locator(".board-mode")).toContainText(`Nước ${afterAI.ply}`);
    await expect(page.getByRole("button", { name: /Chạy AI/ })).toBeDisabled();
    const reset = page.waitForResponse((r) => r.url().endsWith("/reset"));
    await page.getByRole("button", { name: "Đặt lại", exact: true }).click();
    expect((await (await reset).json()).ply).toBe(0);
    await expect(page.locator(".board-mode")).toContainText("Nước 0");
    expect(errors).toEqual([]);
  });
}
