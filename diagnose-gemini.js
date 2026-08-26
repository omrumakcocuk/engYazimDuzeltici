const fs = require("node:fs");
const path = require("node:path");

for (const line of fs.readFileSync(path.join(__dirname, ".env"), "utf8").split(/\r?\n/)) {
  const match = line.match(/^([^#=]+)=(.*)$/);
  if (!match) continue;
  const name = match[1].trim();
  let value = match[2].trim();
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    value = value.slice(1, -1);
  }
  if (!process.env[name]) process.env[name] = value;
}

const key = process.env.GEMINI_API_KEY;
const model = process.env.GEMINI_MODEL || "gemini-3.6-flash";

async function main() {
  if (!key) throw new Error("GEMINI_API_KEY eksik.");

  const modelStarted = Date.now();
  const modelResponse = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}?key=${encodeURIComponent(key)}`,
    { signal: AbortSignal.timeout(15000) }
  );
  const modelData = await modelResponse.json();
  console.log("MODEL_CHECK", modelResponse.status, `${Date.now() - modelStarted}ms`, modelData.error?.message || modelData.name || "");
  if (!modelResponse.ok) return;

  const generateStarted = Date.now();
  const generateResponse = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      signal: AbortSignal.timeout(30000),
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: "Reply with only OK" }] }],
        generationConfig: {
          temperature: 0,
          maxOutputTokens: 32,
          thinkingConfig: { thinkingLevel: "minimal" }
        }
      })
    }
  );
  const generateData = await generateResponse.json();
  console.log(
    "TINY_GENERATE",
    generateResponse.status,
    `${Date.now() - generateStarted}ms`,
    generateData.error?.message || generateData.candidates?.[0]?.finishReason || ""
  );
}

main().catch((error) => {
  console.log("CONNECTION_ERROR", error.name, error.message, error.cause?.code || "");
  process.exitCode = 1;
});
