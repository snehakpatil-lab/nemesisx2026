const fs = require("node:fs");
const path = require("node:path");

const projectRoot = path.resolve(__dirname, "..");
const staticDirectory = path.join(projectRoot, "static");
const outputDirectory = path.join(projectRoot, "dist");
const configuredApiUrl = process.env.NEMESIS_API_BASE_URL;

if (!configuredApiUrl) {
  throw new Error("Set NEMESIS_API_BASE_URL to the deployed Render service origin.");
}

const apiUrl = new URL(configuredApiUrl);
if (
  !["https:", "http:"].includes(apiUrl.protocol) ||
  apiUrl.pathname !== "/" ||
  apiUrl.search ||
  apiUrl.hash ||
  (apiUrl.protocol !== "https:" &&
    !["localhost", "127.0.0.1"].includes(apiUrl.hostname))
) {
  throw new Error("NEMESIS_API_BASE_URL must be an HTTPS origin without a path.");
}

fs.mkdirSync(outputDirectory, { recursive: true });
fs.cpSync(staticDirectory, outputDirectory, { recursive: true });
fs.writeFileSync(
  path.join(outputDirectory, "runtime-config.js"),
  `window.NEMESIS_API_BASE_URL = ${JSON.stringify(apiUrl.origin)};\n`,
  "utf8",
);
