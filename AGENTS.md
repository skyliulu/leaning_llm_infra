# 项目级 Agent 指南

这个仓库是 LLM Infra、RL Infra 与共性系统基础的学习地图。进入项目后，优先把它当成一个持续演进的技术知识库，而不是资料堆放区。

## 仓库定位

- `00-foundations/`：LLM Infra 与 RL Infra 都会依赖的底层基础，包括 GPU / CUDA、计算机系统、分布式系统、深度学习系统和工程工具链。
- `01-llm-infra/`：大模型系统链路，包括数据基础设施、训练系统、推理与服务化、RAG 与 Agent Runtime、评测可观测性、平台治理。
- `02-rl-infra/`：强化学习基础设施，包括环境系统、rollout 与数据管线、训练系统、实验平台、Offline RL 数据集、RLHF / 后训练、安全评测。
- `.agents/skills/`：项目级 Codex 工作流与质量规范。
- `docs/`：MkDocs 站点入口层，通过 Git symlink 暴露根目录中的文档内容。

根目录只保留全局入口和项目配置。具体专题内容应放在对应领域目录下。

## 目录组织

新增主题时，优先使用“领域目录 / 专题目录 / 文章与资源”的结构：

```text
领域目录/
└── topic-name/
    ├── README.md            # 专题导览、阅读顺序、关键问题
    ├── xxx.md               # 正文文章，文件名使用有语义的 snake_case
    ├── assets/              # 图片、图表、附件
    └── examples/            # 可选：代码、配置或实验样例
```

命名约定：

- 目录名使用英文 `kebab-case`，例如 `gpu-architecture`、`inference-serving`。
- Markdown 正文文件使用英文 `snake_case`，例如 `cuda_intro.md`、`rl_theory_foundations.md`。
- `README.md` 只做导览，不承载长篇正文。
- 图片、SVG、示例代码放在同一专题目录下，保持源码仓库和博客站点使用同一套相对路径。
- 如果主题开始变大，再拆 `concepts/`、`papers/`、`projects/`、`notes/`；内容少时先保持扁平。

## 新增内容同步清单

新增技术内容时，不要只创建正文文件。根据改动粒度同步更新入口、导览和站点导航。

### 新增一个领域

例如新增 `03-serving-infra/` 这种顶层方向时，需要同步：

- 根目录 `README.md`：补充项目主线、内容边界和当前重点。
- 新领域目录下的 `README.md`：说明该领域解决什么问题、建议专题、推荐阅读顺序和笔记关注点。
- `mkdocs.yml`：在 `nav` 中增加中文导航项。
- `AGENTS.md`：如果新领域改变了仓库组织规则或写作边界，补充对应说明。
- `docs/`：如果站点仍依赖 Git symlink 暴露根目录内容，需要确认新领域能被 MkDocs 访问。

### 新增一个专题目录

例如在 `02-rl-infra/` 下新增 `rollout-data-pipeline/` 时，需要同步：

- 父目录 `README.md`：在建议专题、当前专题或推荐阅读顺序中加入新专题。
- 新专题 `README.md`：写清楚专题定位、关键问题、文章列表和推荐阅读路径。
- `mkdocs.yml`：如果该专题已有可读正文或导览页，应加入中文导航。
- 专题目录内的 `assets/`、`examples/`：只在确实有图片或代码样例时创建。

### 新增一篇正文文章

例如新增 `rollout_data_pipeline.md` 时，需要同步：

- 所在专题 `README.md`：把文章加入文章列表，并说明它解决的问题。
- 上级领域 `README.md`：如果这篇文章代表一个新的重点方向，补充到当前重点或推荐阅读顺序。
- `mkdocs.yml`：将文章加入站点导航，导航标题使用中文说明。
- 文章中的图片：放在同专题 `assets/`，使用相对路径引用。
- 文章中的代码：短代码可内嵌；可运行样例放在同专题 `examples/`。
- 引用资料：使用公开 GitHub、官方文档、论文或稳定 URL，不使用本地路径。

### 扩展已有文章的小节

如果只是给已有文章新增一节，也要检查：

- 文章目录或开头路线图是否需要更新。
- 所在专题 `README.md` 的文章摘要是否仍准确。
- `mkdocs.yml` 通常不需要更新，除非新增的是独立页面。
- 新增图、公式或代码是否需要进入 `assets/`、`examples/` 或渲染脚本。
- 如果新增内容改变了文章主线，需同步章节小结、过渡段和参考资料。

## 写作规则

长文写作或大幅修改时，使用 `.agents/skills/write-technical-blog/SKILL.md`。

项目内技术文章应尽量回答：

1. 这个模块解决什么系统问题？
2. 它处在 LLM Infra / RL Infra 的哪一层？
3. 输入、输出、状态和核心数据流是什么？
4. 关键设计权衡、瓶颈和失败模式在哪里？
5. 代表性系统、论文、项目或官方资料有哪些？

写作要求：

- 先分析资料和概念主线，再组织正文。
- 术语必须先解释再使用。
- 理论文章应先建立完整案例，再进入抽象公式。
- 同级章节结构要统一；如果一节有“主线”和“小结”，同级章节也应保持同样节奏。
- 核心公式推导要可见，但不能压过核心思想。
- 不要用折叠块隐藏主线内容或关键推导。
- 引用使用公开 GitHub、官方文档、论文或稳定 URL，不引用本地路径。

文章结构校验：

```powershell
python .agents/skills/write-technical-blog/scripts/validate_article.py path/to/article.md
```

## 技术图规则

创建或修改图时，使用 `.agents/skills/design-technical-diagrams/SKILL.md`。

项目内技术图应满足：

- 每张图只回答一个清晰问题。
- 算法流程、数学关系、系统结构优先使用论文风格图，而不是装饰性图片。
- 同一篇文章中的图内语言保持一致。当前文章若已有英文图内标签，新图也应使用英文标签，即使正文是中文。
- 公式不要用普通 SVG 文本模拟；需要出现在图中时，应使用 LaTeX 渲染流程。
- 保留可编辑源文件，例如 `.svg`；文章引用的发布资产优先使用渲染后的 `.png`。
- 修改后必须打开最终渲染图检查文字越界、公式裁剪、箭头遮挡和语言混用。

SVG 结构校验：

```powershell
python .agents/skills/design-technical-diagrams/scripts/validate_svg.py path/to/figure.svg
```

## MkDocs 与 GitHub Pages

博客站点使用 MkDocs Material：

- 主配置：`mkdocs.yml`
- 额外脚本：`docs/assets/mathjax.js`
- 额外样式：`docs/assets/extra.css`
- 公式渲染：`pymdownx.arithmatex` + MathJax
- 目录导航应尽量使用中文说明，匹配中文文章的阅读语境。
- 新增文章后必须更新 `mkdocs.yml` 的 `nav`。
- `navigation.instant` 已开启，MathJax 需要在页面切换和右侧目录 hash 跳转后重新渲染。

Windows PowerShell 本地预览：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m mkdocs serve
```

macOS / Linux 本地预览：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m mkdocs serve
```

注意：`docs/` 下包含 Git symlink。Windows 如果未启用 symlink 支持，本地构建可能提示导航文件不存在；GitHub Actions 的 Linux 环境会正常解析。

## 验证清单

根据改动范围选择验证：

- Markdown 文章变更：运行 `validate_article.py`。
- SVG 变更：运行 `validate_svg.py`，重新渲染 PNG，并逐图检查最终像素。
- MkDocs 配置或站点资源变更：用 MkDocs 加载配置；具备 symlink 支持时运行 `mkdocs build --strict`。
- JavaScript 变更：运行 `node --check path/to/file.js`。
- 通用格式检查：运行 `git diff --check`。
- 结束前查看 `git status --short`，确认没有意外文件。

## Git 协作

- 用户没有要求时，不主动提交或推送。
- 大幅改写或重构一篇正文文章后，完成对应验证并确认工作区范围干净时，应主动提交并推送，避免长文成果只停留在本地。
- 提交应按用户确认的工作范围保持聚焦。
- 不提交 `.venv/`、`site/`、缓存、临时截图、一次性预览图。
- 如果为了验证创建本地环境，放在 `.venv/`，保持未跟踪。
- 遇到用户已有改动时，先读懂并顺着改，不要回滚不属于自己的变更。

## 协作偏好

- 用户偏好可长期复用的项目内产物，而不是只停留在聊天建议。
- 如果某个流程反复出现，应考虑沉淀到 `.agents/skills/`。
- 用户反馈“看不懂”时，优先处理概念连贯性、例子、图解和章节节奏，不是继续堆公式。
- 用户反馈“图有问题”时，必须检查最终渲染结果，不只看 SVG 源码。
