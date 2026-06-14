import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const assetDir = path.resolve(import.meta.dirname, "../assets");
const buildDir = path.join(os.tmpdir(), "rl-theory-diagrams");
const renderDir = path.join(buildDir, "embedded-svg");
const formulaDir = path.join(import.meta.dirname, "formula-cache");
const chrome =
  process.env.CHROME ||
  "C:/Program Files/Google/Chrome/Application/chrome.exe";
const outputScale = 2;

fs.mkdirSync(renderDir, { recursive: true });

function box(latex, cx, cy, width, height) {
  return { latex, cx, cy, width, height };
}

function gridStates(
  offsetX,
  offsetY,
  cellW,
  cellH,
  position = "center",
) {
  const formulas = [];
  let state = 1;
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      const cx =
        position === "corner"
          ? offsetX + col * cellW + 22
          : offsetX + (col + 0.5) * cellW;
      const cy =
        position === "corner"
          ? offsetY + row * cellH + 20
          : offsetY + (row + 0.42) * cellH;
      formulas.push(
        box(
          `s_{${state}}`,
          cx,
          cy,
          position === "corner" ? 36 : 42,
          position === "corner" ? 23 : 28,
        ),
      );
      state += 1;
    }
  }
  return formulas;
}

const diagrams = {
  "actor_critic.svg": [
    box(String.raw`\pi_\theta(a\mid s)`, 235, 204, 190, 34),
    box(String.raw`(s_t,a_t)\rightarrow(r_{t+1},s_{t+1})`, 700, 204, 245, 34),
    box(String.raw`V_w(s)\ \mathrm{or}\ Q_w(s,a)`, 1165, 204, 230, 34),
    box(
      String.raw`\delta_t=r_{t+1}+\gamma V_w(s_{t+1})-V_w(s_t)`,
      1165,
      438,
      325,
      38,
    ),
    box(
      String.raw`\theta\leftarrow\theta+\alpha\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\delta_t`,
      700,
      596,
      555,
      40,
    ),
  ],
  "bellman_backup.svg": [
    box("s", 305, 345, 28, 25),
    box(String.raw`s'`, 405, 345, 32, 25),
    box(String.raw`s'`, 305, 445, 32, 25),
    box(String.raw`s'`, 505, 545, 32, 25),
    box("s", 785, 345, 28, 25),
    box("a", 925, 345, 28, 25),
    box(String.raw`s'_1`, 1095, 235, 42, 27),
    box(String.raw`s'_2`, 1095, 345, 42, 27),
    box(String.raw`s'_3`, 1095, 455, 42, 27),
    box(String.raw`p_1,r_1`, 1002, 250, 60, 18),
    box(String.raw`p_2,r_2`, 1010, 323, 60, 18),
    box(String.raw`p_3,r_3`, 1002, 431, 60, 18),
    box(String.raw`r+\gamma V(s')`, 1305, 360, 155, 34),
    box(
      String.raw`V(s)\leftarrow\mathrm{E}_{\pi,p}[r+\gamma V(s')\mid s]`,
      1070,
      548,
      430,
      40,
    ),
  ],
  "course_policy_returns.svg": [
    ...gridStates(75, 105, 95, 95, "corner"),
    ...gridStates(700, 105, 95, 95, "corner"),
    box(
      String.raw`s_1\rightarrow s_2\rightarrow s_5\rightarrow s_8\rightarrow s_9`,
      245,
      430,
      335,
      34,
    ),
    box(String.raw`G_0=0+0+0+1=1`, 230, 467, 300, 34),
    box(
      String.raw`s_1\rightarrow s_4\rightarrow s_7\rightarrow s_8\rightarrow s_9`,
      870,
      430,
      335,
      34,
    ),
    box(String.raw`G_0=0-1+0+1=0`, 855, 467, 300, 34),
  ],
  "dqn_dataflow.svg": [
    box(String.raw`(s,a,r,s')`, 155, 216, 145, 30),
    box(String.raw`Q(s,a;w)`, 770, 159, 145, 30),
    box(String.raw`Q(s,a;w^-)`, 770, 374, 155, 30),
    box(String.raw`(y-Q(s,a;w))^2`, 1035, 268, 145, 34),
  ],
  "epsilon_greedy.svg": [
    box(String.raw`a_1`, 143, 264, 32, 20),
    box(String.raw`a_2`, 198, 219, 32, 20),
    box(String.raw`a_3`, 253, 249, 32, 20),
    box(String.raw`a^*=\arg\max_a q(s,a)`, 218, 362, 250, 34),
    box(String.raw`1-\varepsilon`, 907, 195, 95, 30),
    box(String.raw`\varepsilon`, 907, 341, 65, 30),
  ],
  "function_approximation.svg": [
    box(String.raw`s_1`, 165, 207, 40, 24),
    box(String.raw`s_2`, 265, 207, 40, 24),
    box(String.raw`s_3`, 365, 207, 40, 24),
    box(String.raw`v_1`, 165, 264, 40, 24),
    box(String.raw`v_2`, 265, 264, 40, 24),
    box(String.raw`v_3`, 365, 264, 40, 24),
    box("s", 700, 265, 28, 24),
    box(String.raw`f_w`, 835, 250, 55, 28),
    box(String.raw`V(s;w)`, 955, 265, 88, 28),
    box(
      String.raw`\min_w\ \mathrm{E}[(V_\pi(S)-V(S;w))^2]`,
      815,
      360,
      345,
      36,
    ),
  ],
  "gpi_loop.svg": [
    box(String.raw`v_\pi=r_\pi+\gamma P_\pi v_\pi`, 330, 316, 280, 36),
    box(
      String.raw`\pi'(s)\in\arg\max_a q_\pi(s,a)`,
      1070,
      316,
      285,
      36,
    ),
  ],
  "mc_td_dp_comparison.svg": [
    box("s", 130, 275, 28, 24),
    box(String.raw`s'`, 245, 220, 32, 24),
    box(String.raw`s'`, 260, 325, 32, 24),
    box("s_0", 505, 275, 38, 24),
    box("s_1", 590, 275, 38, 24),
    box("s_T", 675, 275, 38, 24),
    box(String.raw`\mathrm{target}=G_t`, 600, 385, 180, 30),
    box("s_t", 900, 275, 38, 24),
    box("s_{t+1}", 1025, 275, 52, 24),
    box("r", 965, 240, 18, 20),
    box(
      String.raw`\mathrm{target}=r_{t+1}+\gamma V(s_{t+1})`,
      980,
      385,
      245,
      30,
    ),
  ],
  "mdp_loop.svg": [
    box(String.raw`A_t\sim\pi(\cdot\mid S_t)`, 300, 285, 240, 34),
    box(
      String.raw`(S_{t+1},R_{t+1})\sim p(\cdot\mid S_t,A_t)`,
      1100,
      285,
      300,
      34,
    ),
    box(
      String.raw`G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots`,
      700,
      578,
      440,
      38,
    ),
  ],
  "running_gridworld.svg": [
    ...gridStates(80, 145, 130, 130),
    box("-1", 1370, 243, 45, 25),
    box("-1", 1370, 298, 45, 25),
    box("+1", 1370, 353, 45, 25),
    box("0", 1370, 408, 35, 25),
  ],
  "ppo_dpo_grpo_map.svg": [],
  "rl_storyline.svg": [],
  "bellman_evaluation_control.svg": [],
  "policy_gradient_signal.svg": [],
  "ppo_clip_intuition.svg": [],
  "dpo_pairwise_flow.svg": [],
  "grpo_group_baseline.svg": [],
  "long_horizon_credit.svg": [],
};

function formulaKey(latex) {
  return crypto.createHash("sha1").update(latex).digest("hex").slice(0, 16);
}

function readFormula(latex) {
  const key = formulaKey(latex);
  const pngPath = path.join(formulaDir, `${key}.png`);
  if (!fs.existsSync(pngPath)) {
    throw new Error(
      `Missing cached LaTeX formula ${key}.png for: ${latex}`,
    );
  }
  return fs.readFileSync(pngPath);
}

function pngDimensions(buffer) {
  const signature = buffer.subarray(1, 4).toString("ascii");
  if (signature !== "PNG") {
    throw new Error("Formula cache contains a non-PNG file");
  }
  return {
    width: buffer.readUInt32BE(16),
    height: buffer.readUInt32BE(20),
  };
}

function formulaImage(formula) {
  const png = readFormula(formula.latex);
  const dimensions = pngDimensions(png);
  const scale = Math.min(
    formula.width / dimensions.width,
    formula.height / dimensions.height,
  );
  const width = dimensions.width * scale;
  const height = dimensions.height * scale;
  const x = formula.cx - width / 2;
  const y = formula.cy - height / 2;
  const href = `data:image/png;base64,${png.toString("base64")}`;
  return `<image href="${href}" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${width.toFixed(2)}" height="${height.toFixed(2)}"/>`;
}

function renderDiagram(fileName, formulas) {
  const sourcePath = path.join(assetDir, fileName);
  const outputPath = path.join(assetDir, fileName.replace(/\.svg$/, ".png"));
  const embeddedPath = path.join(renderDir, fileName);
  let svg = fs.readFileSync(sourcePath, "utf8");
  const mathElements = svg.match(
    /<text\b[^>]*class="math"[^>]*>[\s\S]*?<\/text>/g,
  ) ?? [];

  if (mathElements.length !== formulas.length) {
    throw new Error(
      `${fileName}: found ${mathElements.length} math text elements, ` +
        `but ${formulas.length} LaTeX overlays were specified`,
    );
  }

  svg = svg.replace(
    /<text\b[^>]*class="math"[^>]*>[\s\S]*?<\/text>/g,
    "",
  );

  const svgTag = svg.match(/<svg\b[^>]*>/)?.[0] ?? "";
  const sourceWidth = Number(svgTag.match(/\bwidth="(\d+(?:\.\d+)?)"/)?.[1]);
  const sourceHeight = Number(svgTag.match(/\bheight="(\d+(?:\.\d+)?)"/)?.[1]);
  if (!sourceWidth || !sourceHeight) {
    throw new Error(`${fileName}: missing numeric SVG width or height`);
  }
  svg = svg.replace("</svg>", `${formulas.map(formulaImage).join("\n")}\n</svg>`);
  fs.writeFileSync(embeddedPath, svg, "utf8");
  execFileSync(
    chrome,
    [
      "--headless=new",
      "--disable-gpu",
      "--hide-scrollbars",
      `--force-device-scale-factor=${outputScale}`,
      `--window-size=${sourceWidth},${sourceHeight}`,
      `--screenshot=${outputPath}`,
      pathToFileURL(embeddedPath).href,
    ],
    { stdio: "ignore" },
  );
  console.log(`rendered ${path.basename(outputPath)} (${formulas.length} formulas)`);
}

for (const [fileName, formulas] of Object.entries(diagrams)) {
  renderDiagram(fileName, formulas);
}

fs.rmSync(renderDir, { recursive: true, force: true });
