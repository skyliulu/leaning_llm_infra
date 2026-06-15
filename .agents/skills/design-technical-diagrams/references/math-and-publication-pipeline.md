# Mathematical Diagram Publication Pipeline

Use this workflow when a technical figure contains formulas or must render consistently in Markdown, GitHub, IDE previews, and exported documents.

## Classify the artifacts

Keep four categories distinct:

| Category | Example | Commit? |
|---|---|---|
| Editable source | SVG, Graphviz, plotting source | Yes |
| Published asset | High-resolution PNG referenced by the article | Yes |
| Required rebuild input | Font file, formula cache, deterministic manifest | Yes |
| Disposable output | Contact sheet, test screenshot, downloaded runtime, temporary PDF | No |

A file is not temporary merely because it is generated. If the article references it or the checked-in rebuild script requires it, it is a project artifact. Put scratch output in the system temporary directory and remove it after the run.

## Separate prose from mathematics

- Keep explanatory labels as native SVG text.
- Keep symbols, subscripts, superscripts, expectations, gradients, and equations as LaTeX.
- Do not write approximations such as `s1`, `w_T`, `argmax_a`, or Unicode math inside ordinary labels when mathematical typesetting matters.
- Replace mathematical detail with semantic prose when the exact symbol is unnecessary, for example `periodic parameter copy` instead of embedding parameter notation in an arrow label.

For each formula, record:

```text
latex:
center_x:
center_y:
max_width:
max_height:
```

Compile formulas to tightly cropped transparent images, preserve aspect ratio, and fit each image inside its declared box. Hashing the LaTeX source is a practical way to deduplicate a small formula cache.

The renderer must verify that the number of formula placeholders in the source matches the number of formula specifications. A count mismatch is an error, not a warning.

## Design for bounded text

Do not solve overflow by shrinking everything.

1. Shorten the label.
2. Move explanation into the caption.
3. Split the label across deliberate lines.
4. Enlarge or reposition the node.
5. Reduce type size only after the previous options fail.

Use explicit padding. A label or formula should not visually touch a border even if its measured bounding box technically fits.

For grids and trajectories, place state names in a consistent corner when arrows occupy cell centers. For flows, route arrows behind labels or move labels away from the path. Never let an arrow cross a state name, equation, or unrelated annotation.

## Produce the publication PNG

Keep the editable SVG, but publish a PNG when viewer-dependent font metrics or baselines can alter the layout.

Recommended sequence:

1. Remove formula placeholder text from a render copy of the SVG.
2. Embed or composite the LaTeX formula images.
3. Render at 2x the intended logical dimensions.
4. Use an opaque white background unless transparency is explicitly required.
5. Keep output dimensions stable across rebuilds.
6. Point the article at the PNG while retaining the SVG beside it.

The rebuild command should:

- regenerate every publication figure, including figures without formulas;
- fail on missing formula cache entries or placeholder mismatches;
- use a system temporary directory for intermediate files;
- leave the repository clean except for intentional published outputs.

## Validate in layers

Structural validation catches malformed assets; visual validation catches bad figures. Run both.

### Structural checks

- SVG parses and has explicit dimensions and `viewBox`.
- Accessibility metadata and opaque background exist.
- Marker references resolve.
- Every Markdown image reference exists.
- Rebuild script syntax is valid.
- Expected source, PNG, and rebuild-input counts match.

### Pixel checks

Inspect every final PNG at full size and at article width:

- longest prose label;
- widest formula;
- labels changed by translation or terminology cleanup;
- text and formula padding;
- canvas and node boundaries;
- arrowheads and attachment points;
- arrows crossing labels;
- cropped descenders, superscripts, or subscripts;
- contrast and background;
- reading order.

A contact sheet is useful for comparison, but it does not replace opening every revised or high-risk figure at original resolution. Translation, formula rendering, and font substitution are pixel-level risks.

## Common failure patterns

### SVG passes validation but still looks wrong

XML and marker checks cannot detect font substitution, baseline drift, overflow, or semantic confusion. Render and inspect the pixels.

### Formulas look unlike the article

The figure used ordinary SVG text or Unicode symbols while the article used LaTeX. Use the same mathematical typesetting pipeline for both.

### Labels fit in source but overflow in PNG

The rendering engine used different font metrics. Shorten or reflow the label, add explicit padding, and publish a fixed PNG.

### Paths obscure state names

The labels and arrows compete for the same visual channel. Move state labels to cell corners and reserve the center for trajectories.

### The repository contains many mysterious PNGs

Classify them before deletion. Published figures and small deterministic formula caches may be required; contact sheets, test screenshots, palettes, and renderer downloads are disposable.
