import { createHash } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync, unlinkSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { tmpdir } from "node:os";

if (process.env.NEXT_PUBLIC_RESULTS_ONLY === "1") {
  if (!process.env.VERCEL && existsSync("public/data/meta.json") && existsSync("public/video.mp4")) {
    console.log("Using local exported results.");
  } else {
    const url = "https://github.com/SiddhantSwan12/Traffic_Analysis_Agent/releases/download/dashboard-results-v1/dashboard-results.tar.gz";
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Results download failed: ${response.status}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    const digest = createHash("sha256").update(bytes).digest("hex");
    if (digest !== "8fdd99100932805aa696dcc730dba480a1c050ae25964ab9b444f78e1bbf2420") {
      throw new Error("Results archive checksum mismatch");
    }
    const archive = join(tmpdir(), "traffic-dashboard-results.tar.gz");
    writeFileSync(archive, bytes);
    mkdirSync("public", { recursive: true });
    try { execFileSync("tar", ["-xzf", archive, "-C", "public"], { stdio: "inherit" }); }
    finally { unlinkSync(archive); }
    console.log("Verified and unpacked existing traffic analysis.");
  }
}
