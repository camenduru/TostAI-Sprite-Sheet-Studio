// Bridge between the Python editor and spritefusion-pixel-snapper's WASM build.
//
// The editor is server-side numpy, so it cannot call the wasm_bindgen module
// directly. One JSON request arrives on stdin, the package is loaded once, and
// every image in the request is snapped in the same process:
//
//   { pkg, wasm, colors, pixelSize, palette, images:[base64 png, ...] }
//   -> { ok, images:[base64 png, ...] }   or   { ok:false, error }
//
// `pixelSize`/`palette` are null to leave that setting on the snapper's default.
// The image count is not limited here; the caller batches.

import fs from "node:fs";
import { pathToFileURL } from "node:url";

function readStdin() {
  return new Promise((resolve, reject) => {
    const chunks = [];
    process.stdin.on("data", (c) => chunks.push(c));
    process.stdin.on("end", () => resolve(Buffer.concat(chunks)));
    process.stdin.on("error", reject);
  });
}

function fail(message) {
  process.stdout.write(JSON.stringify({ ok: false, error: String(message) }));
  process.exit(1);
}

async function main() {
  const raw = await readStdin();
  let req;
  try {
    req = JSON.parse(raw.toString("utf8"));
  } catch (e) {
    return fail("bad request json: " + e.message);
  }
  if (!req.pkg || !req.wasm) return fail("missing pkg/wasm path");

  let mod;
  try {
    mod = await import(pathToFileURL(req.pkg).href);
    // {module_or_path}: an object with Object.prototype, so the wrapper does
    // not print its "deprecated parameters" warning.
    await mod.default({ module_or_path: fs.readFileSync(req.wasm) });
  } catch (e) {
    return fail("could not load spritefusion-pixel-snapper: " + (e.message || e));
  }

  const out = [];
  for (const b64 of req.images || []) {
    let png;
    try {
      png = mod.process_image(
        Buffer.from(b64, "base64"),
        req.colors ?? null,
        req.pixelSize ?? null,
        req.palette ?? null,
      );
    } catch (e) {
      // wasm_bindgen throws the Rust error string directly.
      return fail(e && e.message ? e.message : e);
    }
    out.push(Buffer.from(png).toString("base64"));
  }
  process.stdout.write(JSON.stringify({ ok: true, images: out }));
}

main().catch((e) => fail(e && e.stack ? e.stack : e));
