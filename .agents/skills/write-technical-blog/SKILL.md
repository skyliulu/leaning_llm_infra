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

Place optional prerequisite material in a compact primer or collapsible section. Do not interrupt the main story with definitions that are not yet needed.

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

### 6. Handle mathematics without losing the reader

- Define every symbol before or immediately after first use.
- State the operation being performed between equations.
- Show intermediate steps whenever a cancellation, distribution change, expectation identity, optimization condition, or approximation matters.
- Distinguish exact equality, estimator, approximation, objective, and implementation heuristic.
- Explain signs, normalization terms, baselines, and hyperparameters in words.
- Follow a long derivation with a small numerical or conceptual example.
- Never use a formula as a substitute for explaining why the result solves the motivating problem.

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
- terms are used before definition;
- sections can be reordered without changing the story;
- examples appear as isolated decorations;
- repeated phrases such as "the slides say" replace direct explanation;
- modern extensions feel appended rather than derived;
- captions do not explain what to notice;
- most paragraphs are equations with only one sentence of prose.

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
