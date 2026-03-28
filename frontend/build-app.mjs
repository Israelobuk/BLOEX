import { build } from "vite";

await build({
  configFile: false,
  cacheDir: ".vite-cache",
});
