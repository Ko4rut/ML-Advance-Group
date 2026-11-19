import { expect, test } from "@playwright/test";

test("loads a JSON game and can step, seek, and autoplay offline", async ({ page }) => {
  await page.goto("/replay");
  await expect(page.getByRole("heading", { name: "Game replay" })).toBeVisible();
  await expect(page.locator(".board-mode")).toContainText("Nước 0 / 0");

  const fixture = {
    game_id: "alphabeta-example",
    initial_side: "red",
    moves: [
      { ply: 1, actor: "alphabeta", action: { source: 54, target: 45 } },
      { ply: 2, actor: "alphabeta", action: { source: 27, target: 36 } },
    ],
    outcome: { status: "truncated", winner: null },
  };

  await page.getByLabel("Tải file ván cờ").setInputFiles({
    name: "test-game.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(fixture)),
  });

  await expect(page.locator(".board-mode")).toContainText("Nước 0 / 2");
  await expect(page.getByText("alphabeta-example").first()).toBeVisible();
  await page.getByRole("button", { name: "Tới một nước" }).click();
  await expect(page.locator(".board-mode")).toContainText("Nước 1 / 2");
  await expect(page.locator(".board-intersection").nth(36)).toHaveAttribute("aria-label", /Tốt đỏ/);

  await page.getByRole("button", { name: "Lùi một nước" }).click();
  await expect(page.locator(".board-mode")).toContainText("Nước 0 / 2");
  await page.getByRole("button", { name: "Đến cuối" }).click();
  await expect(page.locator(".board-mode")).toContainText("Nước 2 / 2");
  await page.getByRole("button", { name: "Về đầu" }).click();
  await page.getByRole("button", { name: "Phát", exact: true }).click();
  await expect(page.locator(".board-mode")).toContainText("Nước 2 / 2", { timeout: 5000 });
  await expect(page.getByRole("button", { name: "Phát", exact: true })).toBeVisible();

  await page.getByRole("button", { name: /a6a5/ }).click();
  await expect(page.locator(".board-mode")).toContainText("Nước 2 / 2");
  await page.getByRole("slider", { name: "Tua đến nước đi" }).fill("1");
  await expect(page.locator(".board-mode")).toContainText("Nước 1 / 2");
});

test("supports multi-game JSONL, coordinate notation and errors", async ({ page }) => {
  await page.goto("/replay");

  const games = [
    { game_id: "alpha", moves: [{ action: { source: 54, target: 45 } }] },
    { game_id: "neural", summary: { result: { winner: "RED", reason: "resign" } }, moves: [{ move: "b2e2", side: "RED" }] },
  ];

  await page.getByLabel("Tải file ván cờ").setInputFiles({
    name: "games.jsonl",
    mimeType: "application/x-ndjson",
    buffer: Buffer.from(games.map(function toLine(game) {
      return JSON.stringify(game);
    }).join("\n")),
  });

  await expect(page.getByLabel("Chọn ván (2)")).toBeVisible();
  await page.getByLabel("Chọn ván (2)").selectOption("1");
  await expect(page.locator(".board-mode")).toContainText("Nước 0 / 1");
  await page.getByRole("button", { name: "Tới một nước" }).click();
  await expect(page.locator(".board-intersection").nth(22)).toHaveAttribute("aria-label", /Pháo đỏ/);
  await expect(page.locator(".replay-outcome")).toContainText("RED");

  await page.getByLabel("Tải file ván cờ").setInputFiles({
    name: "invalid.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({
      initial_board: [1, 2, 3],
      moves: [{ source: 99, target: 1 }],
    })),
  });
  await expect(page.getByRole("alert")).toContainText("90 ô");
});
