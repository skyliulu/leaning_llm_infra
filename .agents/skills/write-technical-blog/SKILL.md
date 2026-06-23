---
name: write-technical-blog
description: Turn books, papers, PDFs, slides, repositories, notes, or code into a rigorous and readable technical blog article. Use when Codex must analyze source material, preserve its conceptual lineage, explain terminology before use, build intuition with a running example, include step-by-step mathematical derivations, add useful diagrams, cite public sources, revise an existing article for clarity, or validate a Markdown technical article before publication.
---

# Write Technical Blog

Produce an article that teaches a coherent mental model rather than compressing sources into a formula catalog.

## Workflow

### 1. Inspect the destination and sources

- Read repository conventions, nearby articles, topic indexes, and asset layout.
- Inventory all source files before drafting.
- Extract the source's chapter sequence, recurring examples, definitions, central equations, and dependencies.
- Prefer primary sources for modern algorithms or claims that extend beyond the supplied material.
- Separate source facts from your own pedagogical reorganization.

For a long source set, create a private source map:

| Source section | Core question | Required concepts | Key result | Useful example |
|---|---|---|---|---|

Do not expose local filesystem paths as article references. Link public GitHub files, official documentation, papers, or stable web pages.

### 2. Establish the reader contract

Before writing, decide:

- target reader and assumed prerequisites;
- what the reader should understand or derive by the end;
- which mathematical ideas need a short primer;
- which terms must be defined before first use;
- one concrete example that can recur through multiple sections.
- the visible rhythm for peer chapters, including the exact labels used for opening roadmaps and closing summaries.

If the subject is abstract, introduce the running example before the formal theory. Make it complete enough to contain the problem, entities, inputs, outputs, objective, and at least two contrasting outcomes.

### 3. Build the conceptual spine

Preserve the source's main lineage, but organize the article by dependency and motivation:

1. What problem are we solving?
2. Why is the current concept insufficient?
3. What new object or assumption resolves that limitation?
4. How is the new result derived?
5. What does each term mean operationally?
6. How does the running example instantiate it?
7. Why does the next section become necessary?

Use transitions to expose causality. Avoid presenting algorithms as an unrelated list.

### 4. Pass the terminology gate

Before a symbol or term first appears:

- give its plain-language meaning;
- state its mathematical role;
- distinguish it from nearby concepts readers may confuse;
- show it in the running example when possible.

Place optional prerequisite material in a compact primer, appendix, or clearly marked aside. Use collapsible sections only when the user or publishing convention explicitly wants them; do not hide concepts or derivations required for the main argument.

A prerequisite primer should review underlying math or computing ideas that help the reader enter the article. It should not become a dictionary of later article terms, and it should not refer forward to concepts that have not yet been motivated.

### 5. Write each technical section

Use this section pattern when appropriate:

1. **Question or limitation**: explain why the section exists.
2. **Intuition**: describe the mechanism without notation.
3. **Formal definition**: introduce symbols and assumptions.
4. **Derivation**: show meaningful intermediate steps.
5. **Interpretation**: explain what every term does.
6. **Worked example or diagram**: make the abstraction observable.
7. **Boundary and transition**: state limitations and motivate the next method.

Vary the prose naturally; do not mechanically label every paragraph with these names.

For a long multi-section tutorial, give every peer chapter the same visible entry and exit:

- begin with a short roadmap that states the question and two or three conceptual moves;
- end with a compact summary that states the result, the mechanism, and the remaining limitation;
- keep these labels and their visual treatment consistent across all peer chapters.

For MkDocs Material articles in this repository, render these recurring entry and exit blocks as visually distinct admonitions rather than ordinary paragraphs or repeated heading levels:

```markdown
!!! abstract "本节主线"

    State the section's question and conceptual moves.

...

!!! summary "本节小结"

    State the result, mechanism, and remaining limitation.
```

Place `本节主线` immediately after the peer chapter heading and `本节小结` at the end of that peer chapter. Do not leave the roadmap after opening body paragraphs, and do not mix callout-style roadmaps with heading-style summaries in the same article. Avoid using repeated `### 本节主线` / `### 本节小结` headings when the site already uses a right-side table of contents, because they clutter navigation and visually merge with normal section structure.

Do not add a roadmap or summary to only a few chapters. Structural inconsistency makes readers repeatedly relearn how to navigate the article.

When revising an existing draft, first audit all peer chapters for structural drift. Either apply the pattern everywhere or remove it everywhere; partial consistency is worse than a simpler structure.

### 6. Handle mathematics without losing the reader

- Use LaTeX math for every mathematical expression, both inline and display. Do not write formulas in backticks, plain text code blocks, ASCII approximations, or ordinary SVG text. Use `$...$` for inline symbols and `$$...$$` for display equations in Markdown articles.
- Define every symbol before or immediately after first use.
- State the operation being performed between equations.
- Show intermediate steps whenever a cancellation, distribution change, expectation identity, optimization condition, or approximation matters.
- Distinguish exact equality, estimator, approximation, objective, and implementation heuristic.
- Explain signs, normalization terms, baselines, and hyperparameters in words.
- Follow a long derivation with a small numerical or conceptual example.
- Never use a formula as a substitute for explaining why the result solves the motivating problem.
- Keep required derivations visible in the normal reading flow. Do not hide core mathematics inside collapsible blocks by default.
- Make the conceptual claim visually prominent before the derivation, then restate its operational meaning afterward. Formulas should support the idea rather than compete with it for attention.
- If a derivation is long, split it with meaningful subheadings and prose checkpoints. Do not use folding as a substitute for explanation.
- After a dense equation sequence, add a short "what changed" explanation: what was transformed, what assumption was used, and what the reader should now believe.

Read [references/editorial-checklist.md](references/editorial-checklist.md) for the detailed math and narrative review.

### 7. Add figures with a teaching purpose

Use `$design-technical-diagrams` for technical diagrams.

Every figure must answer a named question, such as:

- What interacts with what?
- What information flows through the algorithm?
- Which quantity is updated?
- How do two methods differ?
- Where does a derivation branch?

Introduce the figure in prose and interpret it afterward. A caption should state the takeaway, not merely repeat the title.

### 8. Extend beyond the supplied source carefully

When adding newer methods:

- connect each method to the last established concept;
- state whether it is a continuation, specialization, approximation, or different training paradigm;
- use primary papers or official documentation;
- avoid implying a simple replacement chain when methods differ in data source, online/offline behavior, objective, or system cost;
- add a comparison table only after each method has been explained independently.

### 9. Revise for continuity

Rewrite when the draft shows these symptoms:

- formulas arrive before the reader knows the problem;
- core ideas are visually weaker than the equations that support them;
- a reader must parse every displayed equation before seeing the section's claim;
- terms are used before definition;
- sections can be reordered without changing the story;
- examples appear as isolated decorations;
- repeated phrases such as "the slides say" replace direct explanation;
- modern extensions feel appended rather than derived;
- captions do not explain what to notice;
- most paragraphs are equations with only one sentence of prose.
- peer chapters use different roadmap, summary, or emphasis conventions.
- core formulas or important explanations are hidden in collapsible blocks.
- figures switch language or labeling conventions mid-article.

State established knowledge directly. Attribute specific claims and sources, not every paragraph.

### 10. Validate before finishing

Run:

```powershell
python .agents/skills/write-technical-blog/scripts/validate_article.py path/to/article.md
```

Then manually verify:

- all figures render and remain legible;
- equations render in the target Markdown environment;
- source links are public and reachable;
- the title, abstract, table of contents, headings, summary, and references agree;
- the running example remains consistent;
- no temporary previews or unused assets remain.

The validator catches structural issues only. Passing it does not prove the article is understandable.
