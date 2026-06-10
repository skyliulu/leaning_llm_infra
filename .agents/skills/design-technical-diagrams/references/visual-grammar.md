# Technical Figure Visual Grammar

## Composition

- Prefer a single reading direction.
- Use a grid for alignment; do not position boxes by eye alone.
- Leave at least one text-line of whitespace between groups.
- Keep primary nodes similar in size unless size itself carries meaning.
- Place legends close to the encoding they explain.

## Typography

- Use a broadly available sans-serif font for labels.
- Use a serif math font only for formulas.
- Use no more than four text levels.
- Keep labels concise and use zero letter spacing.
- Avoid tiny footnotes inside figures; move them to captions.

## Color

- Start in grayscale and add color only for semantic grouping.
- Use pale fills with dark text.
- Reserve strong color for one focal state, warning, or selected path.
- Keep online/offline, control/data, or positive/negative encodings consistent.
- Never depend on color alone; also use labels, line styles, or grouping.

## Shapes and Lines

- Use rectangles for components, rounded only slightly.
- Use circles for states or repeated atomic entities when appropriate.
- Use solid arrows for direct flow or dependency.
- Use dashed arrows for optional, inferred, reformulated, or indirect relations.
- Label arrows when direction alone is ambiguous.
- Keep stroke widths and arrowheads consistent.

## Diagram Types

### Process Flow

Show ordered stages and feedback loops. Make the loop visually explicit rather than placing stages in a circle without direction.

### Architecture

Separate control flow, data flow, storage, and external systems. Use lanes or grouped bands when they coexist.

### Algorithm Comparison

Use aligned columns and identical row dimensions. Compare methods using the same vocabulary.

### Mathematical Dependency Map

Show which definition or identity enables each derivation. Keep full equations outside the map unless the formula is the node.

### State or Environment Illustration

Include start, goal, invalid or penalized states, action set, reward semantics, and at least one trajectory when those details matter.

## Caption Pattern

Use:

> Figure N: [what is represented]. [The contrast, flow, or conclusion the reader should notice].

Avoid:

> Figure N: Overview diagram.

## Visual QA

- Render at final width and at 50% scale.
- Check the longest label, not only average labels.
- Verify no text crosses a box edge.
- Verify lines do not pass through unrelated labels.
- Verify arrowheads remain visible against fills.
- Verify the background is opaque when the article viewer may use dark mode.
- Verify raster images are not blurry at their displayed size.
