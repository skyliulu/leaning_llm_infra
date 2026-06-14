# RL Theory Foundations

这个目录整理强化学习基础理论主线，重点是从 MDP、Bellman 方程、动态规划、Monte Carlo、TD、函数近似一路走到 policy gradient、actor-critic、PPO、DPO 和 GRPO。

## 文章

- [强化学习基础理论：从 Bellman 方程到 PPO、DPO 与 GRPO](./rl_theory_foundations.md)

## 图片资产

正文图片放在 [assets](./assets/) 中。其中 `rl_theory_roadmap_imagegen.png` 使用 imagegen 生成；技术图保留 SVG 作为可编辑源文件，并导出固定渲染的高分辨率 PNG 供正文引用，避免不同查看器的字体与文本基线差异破坏布局。

技术图中的数学符号由 LaTeX 透明图层合成，缓存位于 `scripts/formula-cache/`。在 Windows 上运行 `node scripts/render_latex_diagrams.mjs` 可用无头 Chrome 重新生成全部技术图 PNG。
