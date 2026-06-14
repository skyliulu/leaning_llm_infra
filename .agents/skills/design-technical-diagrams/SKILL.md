---
name: design-technical-diagrams
description: Design, generate, edit, and validate clear technical figures for blogs, papers, documentation, and teaching material. Use when Codex must create paper-style architecture diagrams, algorithm flows, mathematical dependency maps, comparison figures, state-transition illustrations, or conceptual images; choose between SVG, Mermaid, Graphviz, plotting, and image generation; or revise a figure that is confusing, decorative, cartoonish, unreadable, or visually unverified.
---

# Design Technical Diagrams

Create figures that reduce explanation cost. Treat visual correctness and legibility as part of technical correctness.

## Start with the message

Write one sentence before drawing:

> After seeing this figure, the reader should understand ___.

If the blank requires "and" more than once, split the figure.

Identify:

- the question answered by the figure;
- the entities and relationships that must be visible;
- the reading direction;
- the one element that deserves visual emphasis;
- what detail belongs in the caption rather than the canvas.

Before drawing, inspect the article's existing figure set and choose one canvas language. Use the same language in every figure unless the user explicitly requests bilingual figures. When extending an existing article, follow its established figure language; if no convention exists, default to English for technical labels and keep surrounding captions in the article language.

## Choose the representation

Use the simplest medium that preserves correctness:

| Need | Preferred representation |
|---|---|
| Exact boxes, arrows, states, formulas, or method comparison | SVG, Mermaid, Graphviz, or code-native drawing |
| Quantitative curves, distributions, or measurements | plotting library |
| Spatial scene, physical intuition, cover art, or conceptual atmosphere | image generation |
| Existing bitmap correction or transformation | image editing |

Do not use image generation for a topology or derivation that requires exact labels and edges. Do not use SVG merely to imitate an illustration when a bitmap better communicates the concept.

For generated raster images, use the `imagegen` skill/tool and request a restrained academic or editorial style. Reject cartoon rendering unless the audience or user explicitly wants it.

## Build a semantic specification

Before implementation, list:

```text
Title:
Teaching goal:
Nodes:
Edges:
Groups:
Reading order:
Visual encoding:
Caption takeaway:
```

For comparison figures, align every method on the same dimensions, such as:

- data source;
- objective or learning signal;
- online versus offline sampling;
- model components;
- baseline or critic;
- system cost or limitation.

Do not imply inheritance with an arrow when the relationship is only conceptual. Use dashed arrows and explicit labels for reformulations, optional paths, or weaker relationships.

## Apply paper-style visual grammar

Read [references/visual-grammar.md](references/visual-grammar.md) when creating or revising a figure.

Core defaults:

- white, opaque background;
- near-black text and strokes;
- restrained functional color, not a single-hue wash;
- rectangular groups with small corner radii;
- consistent stroke weight and arrowheads;
- left-to-right or top-to-bottom reading order;
- short labels with explanations outside the node;
- generous whitespace and stable alignment;
- one visual hierarchy: title, group heading, node label, annotation.
- one canvas language across the complete article or figure series.

Avoid decorative gradients, glow, shadows, mascots, floating shapes, pseudo-3D boxes, and unexplained icons.

## Connect the figure to the article

- Introduce the figure before it appears.
- Use terminology already defined in the prose.
- Keep formulas in the article when they would make the figure crowded.
- Write a caption that states what readers should notice.
- Avoid duplicating entire paragraphs inside the canvas.

## Implement exact diagrams

When a diagram contains mathematical notation or will be published through Markdown/GitHub, read [references/math-and-publication-pipeline.md](references/math-and-publication-pipeline.md).

For SVG:

- set explicit `width`, `height`, and `viewBox`;
- include `role="img"` and a meaningful `aria-label`;
- add an opaque background rectangle;
- define reusable styles, markers, and colors;
- keep text inside stable boxes;
- avoid relying on a rare local font;
- use XML-valid escaping;
- keep all arrows visually attached to their intended nodes.

For Markdown articles viewed across GitHub, IDEs, and operating systems, keep SVG as the editable source but prefer a rendered high-resolution PNG as the published asset. This prevents font substitution, text-baseline differences, and viewer-specific SVG behavior from changing the layout.

Do not convert a flawed SVG blindly: fix the source, render it, inspect the pixels, then update the article to reference the PNG. Separate files into editable sources, published assets, required rebuild inputs, and disposable previews; commit only the first three.

Do not imitate mathematical notation with ordinary SVG text. Keep prose as native diagram text and render formulas from LaTeX. Give every formula an explicit placement box and verify that the final rendered formula stays clear of node edges, labels, and arrows.

Run:

```powershell
python .agents/skills/design-technical-diagrams/scripts/validate_svg.py path/to/figure.svg
```

## Render and inspect

Never stop after XML parsing or successful generation.

1. Render the final asset to a bitmap or open it in a browser.
2. Inspect the actual pixels at the intended article width.
3. Inspect every final figure, not only a contact sheet or representative sample.
4. Check text overflow, formula padding, overlap, contrast, arrow attachment, cropping, reading order, and accidental language mixing.
5. Check both light and dark viewers when transparency is possible.
6. Revise until the meaning is clear without reading surrounding implementation code.
7. Re-run structural validation and article image-reference validation.
8. Delete temporary previews and confirm the rebuild leaves no repository-local scratch files.

Use structural validation and visual inspection together. Neither replaces the other.

## Diagnose common failures

### The figure is attractive but unclear

- Reduce it to one question.
- Replace decorative imagery with explicit entities and labeled relationships.
- Add a reading direction.
- Move secondary prose into the caption.

### The figure feels cartoonish

- Remove characters, exaggerated perspective, bubbles, and saturated palettes.
- Use an editorial or academic layout with measured spacing and functional color.
- Prefer exact SVG for algorithm relationships.

### The figure is technically correct but dense

- Split overview and detail into separate figures.
- Group repeated elements.
- Replace sentences inside boxes with short labels.
- Align comparable items into columns or rows.

### The arrows tell the wrong story

- Distinguish chronology, data flow, dependency, optimization, and analogy.
- Label non-obvious arrows.
- Use dashed lines for indirect relationships.
- Avoid arranging methods as a replacement chain unless that claim is true.

## Final acceptance

A figure is finished only when:

- its teaching goal can be stated in one sentence;
- every visible element supports that goal;
- labels are legible at article width;
- formulas use standard mathematical typesetting;
- no text or formula crosses a node or canvas boundary;
- no arrow passes through an unrelated label;
- colors have semantic meaning;
- all figures use the agreed canvas language consistently;
- arrows and grouping encode the intended relationship;
- the caption explains the takeaway;
- every rendered output has been visually inspected;
- published assets can be rebuilt without relying on uncommitted temporary files.
