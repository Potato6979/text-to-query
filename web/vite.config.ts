import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  cacheDir: "../work/vite-cache/web",
  plugins: [react()],
  server: {
    port: 3000,
  },
});
