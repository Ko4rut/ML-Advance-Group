import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  use: { baseURL: process.env.XIANGQI_FRONTEND_URL || "http://127.0.0.1:3000" },
});
