import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    exclude: process.env.E2E ? [] : ["test/e2e.test.ts"],
  },
});
