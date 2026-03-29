import { build } from "vite";
import react from "@vitejs/plugin-react";

await build({
  configFile: false,
  cacheDir: ".vite-cache",
  plugins: [react()],
});
